rm(list=ls(all=TRUE))
setwd("/local/tli89/PineTreeDelineation")

# Load required libraries
library(tools)
library(lidR)
library(future)
library(RCSF)
library(terra)
library(tidyverse)
library(dbscan)
library(sf)



# --- Setup ---
make_plot = TRUE

LASfile_dir <- "data/Rolleston_trial/rolleston_forest_plots" 
LASfile_name <- "plot_1.las"


LASfile_base <- file_path_sans_ext(LASfile_name)
outpath <- file.path(paste0("outputs/", LASfile_base, "_outputs"))

# --- Clean output directory if it exists ---
if (dir.exists(outpath)) {
  unlink(outpath, recursive = TRUE, force = TRUE)
}
dir.create(outpath, recursive = TRUE, showWarnings = FALSE)

output_file <- file.path(outpath, paste0(LASfile_base, "_summary.txt"))
make_plot = TRUE
print(outpath)
print(output_file)

# --- Read and filter LAS ---
las <- readLAS(file.path(LASfile_dir, LASfile_name))
las <- filter_duplicates(las)
las_check(las)
plot(las)
start_time = Sys.time()

ctg = readLAScatalog(file.path(LASfile_dir, LASfile_name))
if (make_plot) plot(ctg)

plan(multisession, workers=3L)
opt_chunk_buffer(ctg) <- 10
opt_chunk_size(ctg) <- 250
opt_filter(ctg) <- ""
opt_output_files(ctg) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_tiled")
newctg <- catalog_retile(ctg)
if (make_plot) plot(newctg, chunk=TRUE)

# --- Ground and noise classification ---
opt_output_files(newctg) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_classifyground")
classified_ctg1 <- classify_ground(newctg, csf(class_threshold = 0.25, cloth_resolution = 0.25, rigidness = 2))

opt_output_files(classified_ctg1) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_classifynoise")
classified_ctg2 <- classify_noise(classified_ctg1, ivf(5, 6))
if (make_plot) plot(classified_ctg2, chunk=TRUE)



# --- DTM creation and smoothing ---
opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_dtm")
rasterize_terrain(classified_ctg2, res = 0.1, tin(), overwrite=TRUE)

dtm_tiles <- list.files(path = outpath, pattern = '_dtm.tif$', full.names = T)
dtm_mosaic <- vrt(dtm_tiles, overwrite = T)
writeRaster(dtm_mosaic, filename = file.path(outpath, paste0(LASfile_base, "_dtm_mosaic.tif")), overwrite = TRUE)

dtm_smooth <- dtm_mosaic %>% focal(w = matrix(1, 25, 25), fun = mean, na.rm = TRUE, pad = TRUE)
writeRaster(dtm_smooth, filename = file.path(outpath, paste0(LASfile_base, '_dtm_mosaic_smooth.tif')), overwrite = TRUE)

if (make_plot) plot(dtm_mosaic, bg = "white")
dtm_prod <- terra::terrain(dtm_mosaic, v = c("slope", "aspect"), unit = "radians")
dtm_hillshade <- terra::shade(slope = dtm_prod$slope, aspect = dtm_prod$aspect)
if (make_plot) plot(dtm_hillshade, col = gray(0:50/50), legend = FALSE)

remove(dtm_mosaic)
remove(dtm_smooth)

# --- Normalize point cloud ---
opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_norm_tin")
opt_filter(classified_ctg2) <- '-drop_as_withheld'
ctg_norm_tin <- normalize_height(classified_ctg2, tin())

plot(ctg_norm_tin)
# --- Canopy Height Model (CHM) ---
opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_chm")
rasterize_canopy(classified_ctg2, res = 0.1, algorithm = p2r(na.fill = knnidw()))

chm_tiles <- list.files(path = outpath, pattern = '_chm.tif$', full.names = T)
chm_mosaic <- vrt(chm_tiles, overwrite = T)
chm_mosaic <- clamp(chm_mosaic, 0, 30, values = TRUE)
names(chm_mosaic) <- 'Z'

chm_filled <- focal(chm_mosaic, w = 3, fun = "mean", na.policy = "only", na.rm = TRUE)
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

chm_smooth <- terra::focal(chm_filled, w = fgauss(2, n = 7))
names(chm_smooth) <- 'Z'

if (make_plot) {
  plot(chm_mosaic, col = height.colors(50))
  plot(chm_filled, col = height.colors(50))
  plot(chm_smooth, col = height.colors(50))
}

writeRaster(chm_mosaic, filename = file.path(outpath, paste0(LASfile_base, '_chm_mosaic.tif')), overwrite = T)
writeRaster(chm_filled, filename = file.path(outpath, paste0(LASfile_base, '_chm_filled.tif')), overwrite = T)
writeRaster(chm_smooth, filename = file.path(outpath, paste0(LASfile_base, '_chm_smooth.tif')), overwrite = T)

# --- DSM ---
opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_dsm")
rasterize_canopy(classified_ctg2, res = 0.1, algorithm = p2r(na.fill = knnidw()))
dsm_tiles <- list.files(path = outpath, pattern = '_dsm.tif$', full.names = T)
dsm_mosaic <- vrt(dsm_tiles, overwrite = T)
names(dsm_mosaic) <- 'Z'
writeRaster(dsm_mosaic, filename = file.path(outpath, paste0(LASfile_base, '_dsm_mosaic.tif')), overwrite = T)

dsm_filled <- terra::focal(dsm_mosaic, w = 3, fun = "mean", na.policy = "only", na.rm = TRUE)
dsm_smooth <- terra::focal(dsm_filled, w = fgauss(1, n = 5))
names(dsm_smooth) <- 'Z'

writeRaster(dsm_filled, filename = file.path(outpath, paste0(LASfile_base, '_dsm_filled.tif')), overwrite = T)
writeRaster(dsm_smooth, filename = file.path(outpath, paste0(LASfile_base, '_dsm_smooth.tif')), overwrite = T)

if (make_plot) {
  plot(dsm_mosaic, col = height.colors(50))
  plot(dsm_filled, col = height.colors(50))
  plot(dsm_smooth, col = height.colors(50))
}

# --- Tree top detection using DBSCAN on top 4 m ---

# Read the filtered, classified LAS files (after noise classification)
classified_files <- list.files(path = outpath, pattern = "_classifynoise.las$", full.names = TRUE)
las_clean <- readLAS(classified_files)

# Remove any remaining noise-classified points (classification == 18 in LAS format)
las_clean <- filter_poi(las_clean, Classification != 18)

# Compute height threshold for top 4 meters
max_z <- max(las_clean@data$Z, na.rm = TRUE)
top_threshold <- max_z - 4
las_top <- filter_poi(las_clean, Z >= top_threshold)
# Prepare XYZ data for clustering
xyz <- as.data.frame(las_top@data[, c("X", "Y", "Z")])
plot(las_top)
# DBSCAN clustering
eps_val <- 0.4
minpts_val <- 20
db <- dbscan(xyz[, c("X", "Y")], eps = eps_val, minPts = minpts_val)


# Assign cluster IDs and keep non-noise clusters
xyz$cluster <- db$cluster
xyz <- xyz[xyz$cluster != 0, ]

# Rename and convert to numeric
xyz$treeID <- as.numeric(xyz$cluster)
xyz$cluster <- NULL  # optional: remove old column

# Extract the highest point per tree
ttops_dbscan <- xyz %>%
  group_by(treeID) %>%
  slice_max(order_by = Z, n = 1, with_ties = FALSE) %>%
  ungroup()


# Convert to sf and write to shapefile
ttops_sf <- st_as_sf(ttops_dbscan, coords = c("X", "Y"), crs = st_crs(las_clean))
st_write(ttops_sf, dsn = file.path(outpath, paste0(LASfile_base, "_treetops_dbscan.shp")), delete_layer = TRUE)

if (make_plot) plot(ttops_sf)


# --- Tree segmentation ---
opt_output_files(ctg_norm_tin) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_segmented")
algo <- dalponte2016(chm_mosaic, ttops_sf, max_cr = 20, th_cr = 0.8, th_tree = 1)
ctg_segmented <- segment_trees(ctg_norm_tin, algo)

ctg_final = readLAScatalog(outpath, pattern = '_segmented.las')
las = readLAS(ctg_final)
if (make_plot) {plot(las, bg = "white", size = 4, color = "treeID")}

# --- Save summary ---
end_time = Sys.time()
elapsed_seconds <- as.numeric(end_time - start_time, units = "secs")

cat("Total Tree Tops Detected: ", nrow(ttops_sf), "\n")
print(elapsed_seconds)
summary(las)
