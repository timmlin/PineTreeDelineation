rm(list=ls(all=TRUE))

library(tools)
library(lidR)
library(future)
library(RCSF)
library(terra)
library(tidyverse)
library(sf)

# --- User Parameters ---

# Processing parameters
num_workers <- 3
chunk_buffer <- 10
chunk_size <- 250

# Classification parameters
csf_class_threshold <- 0.25
csf_cloth_resolution <- 0.25
csf_rigidness <- 2
ivf_param1 <- 5
ivf_param2 <- 6

# Rasterization
raster_res <- 0.1

# DTM/CHM/DSM smoothing
dtm_smooth_window <- 25
chm_filled_window <- 3
chm_smooth_sigma <- 2
chm_smooth_n <- 7
dsm_filled_window <- 3
dsm_smooth_sigma <- 1
dsm_smooth_n <- 5

# CHM/DSM clamp
chm_clamp_min <- 0
chm_clamp_max <- 30

# Tree detection
lmf_window <- 4
lmf_hmin <- 8.2
lmf_shape <- "circular"

# Dalponte2016 parameters
dalponte_max_cr <- 20
dalponte_th_cr <- 0.8
dalponte_th_tree <- 1

# Plotting
plot_colors <- height.colors(50)

# --- Workflow ---

LASfile_dir <- "path-to-input-las-file"
LASfile_path <- file.path(LASfile_dir, pattern = "input-las-file-name.las")

LASfile_name <- basename(LASfile_path)
LASfile_base <- file_path_sans_ext(LASfile_name)
outpath <- file.path("data/results/dalponte/segmented_las", paste0(LASfile_base, "_outputs"))
if (dir.exists(outpath)) unlink(outpath, recursive = TRUE, force = TRUE)
dir.create(outpath, recursive = TRUE, showWarnings = FALSE)

las <- readLAS(LASfile_path)
las <- filter_duplicates(las)
las_check(las)
rm(las)

ctg <- readLAScatalog(LASfile_path)
plot(ctg)

plan(multisession, workers = num_workers)
opt_chunk_buffer(ctg) <- chunk_buffer
opt_chunk_size(ctg) <- chunk_size
opt_filter(ctg) <- ""
opt_output_files(ctg) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_tiled")
newctg <- catalog_retile(ctg)
plot(newctg, chunk = TRUE)

# Ground and noise classification
opt_output_files(newctg) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_classifyground")
classified_ctg1 <- classify_ground(newctg, csf(class_threshold = csf_class_threshold, cloth_resolution = csf_cloth_resolution, rigidness = csf_rigidness))

opt_output_files(classified_ctg1) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_classifynoise")
classified_ctg2 <- classify_noise(classified_ctg1, ivf(ivf_param1, ivf_param2))
plot(classified_ctg2, chunk = TRUE)

# DTM creation and smoothing
opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_dtm")
rasterize_terrain(classified_ctg2, res = raster_res, tin(), overwrite = TRUE)

dtm_tiles <- list.files(path = outpath, pattern = '_dtm.tif$', full.names = TRUE)
dtm_mosaic <- vrt(dtm_tiles, overwrite = TRUE)
writeRaster(dtm_mosaic, file.path(outpath, paste0(LASfile_base, "_dtm_mosaic.tif")), overwrite = TRUE)

dtm_smooth <- focal(dtm_mosaic, w = matrix(1, dtm_smooth_window, dtm_smooth_window), fun = mean, na.rm = TRUE, pad = TRUE)
writeRaster(dtm_smooth, file.path(outpath, paste0(LASfile_base, "_dtm_mosaic_smooth.tif")), overwrite = TRUE)

plot(dtm_mosaic, bg = "white")
dtm_prod <- terrain(dtm_mosaic, v = c("slope", "aspect"), unit = "radians")
dtm_hillshade <- shade(dtm_prod$slope, dtm_prod$aspect)
plot(dtm_hillshade, col = gray(0:50/50), legend = FALSE)

# Normalize point cloud
opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_norm_tin")
opt_filter(classified_ctg2) <- "-drop_as_withheld"
ctg_norm_tin <- normalize_height(classified_ctg2, tin())
plot(ctg_norm_tin)

# CHM generation
opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_chm")
rasterize_canopy(classified_ctg2, res = raster_res, algorithm = p2r(na.fill = knnidw()))
chm_tiles <- list.files(path = outpath, pattern = '_chm.tif$', full.names = TRUE)
chm_mosaic <- vrt(chm_tiles, overwrite = TRUE)
chm_mosaic <- clamp(chm_mosaic, chm_clamp_min, chm_clamp_max, values = TRUE)
names(chm_mosaic) <- 'Z'

chm_filled <- focal(chm_mosaic, w = chm_filled_window, fun = "mean", na.policy = "only", na.rm = TRUE)
names(chm_filled) <- 'Z'

fgauss <- function(sigma, n = 7) {
  m <- matrix(ncol = n, nrow = n)
  col <- rep(1:n, n)
  row <- rep(1:n, each = n)
  x <- col - ceiling(n / 2)
  y <- row - ceiling(n / 2)
  m[cbind(row, col)] <- 1 / (2 * pi * sigma^2) * exp(-(x^2 + y^2) / (2 * sigma^2))
  m / sum(m)
}

chm_smooth <- focal(chm_filled, w = fgauss(chm_smooth_sigma, chm_smooth_n))
names(chm_smooth) <- 'Z'

plot(chm_mosaic, col = plot_colors)
plot(chm_filled, col = plot_colors)
plot(chm_smooth, col = plot_colors)

writeRaster(chm_mosaic, file.path(outpath, paste0(LASfile_base, "_chm_mosaic.tif")), overwrite = TRUE)
writeRaster(chm_filled, file.path(outpath, paste0(LASfile_base, "_chm_filled.tif")), overwrite = TRUE)
writeRaster(chm_smooth, file.path(outpath, paste0(LASfile_base, "_chm_smooth.tif")), overwrite = TRUE)

# DSM generation
opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_dsm")
rasterize_canopy(classified_ctg2, res = raster_res, algorithm = p2r(na.fill = knnidw()))
dsm_tiles <- list.files(path = outpath, pattern = '_dsm.tif$', full.names = TRUE)
dsm_mosaic <- vrt(dsm_tiles, overwrite = TRUE)
names(dsm_mosaic) <- 'Z'
writeRaster(dsm_mosaic, file.path(outpath, paste0(LASfile_base, "_dsm_mosaic.tif")), overwrite = TRUE)

dsm_filled <- focal(dsm_mosaic, w = dsm_filled_window, fun = "mean", na.policy = "only", na.rm = TRUE)
dsm_smooth <- focal(dsm_filled, w = fgauss(dsm_smooth_sigma, dsm_smooth_n))
names(dsm_smooth) <- 'Z'

writeRaster(dsm_filled, file.path(outpath, paste0(LASfile_base, "_dsm_filled.tif")), overwrite = TRUE)
writeRaster(dsm_smooth, file.path(outpath, paste0(LASfile_base, "_dsm_smooth.tif")), overwrite = TRUE)

plot(dsm_mosaic, col = plot_colors)
plot(dsm_filled, col = plot_colors)
plot(dsm_smooth, col = plot_colors)

# Tree top detection and segmentation
ttops <- locate_trees(ctg_norm_tin, lmf(lmf_window, hmin = lmf_hmin, shape = lmf_shape))

ttops_tiles <- list.files(path = outpath, pattern = '*shp', full.names = T)
ttops_list <- lapply(ttops_tiles, read_sf)
all_ttops <- do.call(rbind, ttops_list)

num_ttops <- seq(from=1, to=nrow(all_ttops))
all_ttops$treeID <- num_ttops
plot(all_ttops)

opt_output_files(ctg_norm_tin) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_segmented")
algo <- dalponte2016(chm_mosaic, all_ttops, max_cr = dalponte_max_cr, th_cr = dalponte_th_cr, th_tree = dalponte_th_tree)
ctg_segmented <- segment_trees(ctg_norm_tin, algo)

ctg_final = readLAScatalog(outpath, pattern = '_segmented.las')
las = readLAS(ctg_final)

plot(las, bg = "white", size = 4, color = "treeID")
  



