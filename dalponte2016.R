rm(list=ls(all=TRUE))
setwd("/local/tli89/PineTreeDelineation")

library(tools) # Load tools package for file path manipulation
library(lidR)
library(future)
library(RCSF)
library(terra)
library(tidyverse)

# --- Parameters ---
make_plot <- FALSE

LASfile_dir <- "data/Rolleston_trial/rolleston_forest_plots"
LASfile_path <- file.path(LASfile_dir, pattern = "plot_1.las")

output_file <- file.path("summary.txt")
file.create(output_file)

# --- Processing Loop ---

  
  LASfile_name <- basename(LASfile_path)
  LASfile_base <- file_path_sans_ext(LASfile_name)
  outpath <- file.path("data/results/dalponte/segmented_las", paste0(LASfile_base, "_outputs"))
  # Clean and create output directory
  if (dir.exists(outpath)) unlink(outpath, recursive = TRUE, force = TRUE)
  dir.create(outpath, recursive = TRUE, showWarnings = FALSE)
  

  # Read and filter LAS
  las <- readLAS(LASfile_path)
  las <- filter_duplicates(las)
  las_check(las)
  rm(las)
  
  start_time <- Sys.time()
  
  ctg <- readLAScatalog(LASfile_path)
  if (make_plot) plot(ctg)
  
  plan(multisession, workers = 3L)
  opt_chunk_buffer(ctg) <- 10
  opt_chunk_size(ctg) <- 250
  opt_filter(ctg) <- ""
  opt_output_files(ctg) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_tiled")
  newctg <- catalog_retile(ctg)
  if (make_plot) plot(newctg, chunk = TRUE)
  
  # Ground and noise classification
  opt_output_files(newctg) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_classifyground")
  classified_ctg1 <- classify_ground(newctg, csf(class_threshold = 0.25, cloth_resolution = 0.25, rigidness = 2))
  
  opt_output_files(classified_ctg1) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_classifynoise")
  classified_ctg2 <- classify_noise(classified_ctg1, ivf(5, 6))
  if (make_plot) plot(classified_ctg2, chunk = TRUE)
  
  # DTM creation and smoothing
  opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_dtm")
  rasterize_terrain(classified_ctg2, res = 0.1, tin(), overwrite = TRUE)
  
  dtm_tiles <- list.files(path = outpath, pattern = '_dtm.tif$', full.names = TRUE)
  dtm_mosaic <- vrt(dtm_tiles, overwrite = TRUE)
  writeRaster(dtm_mosaic, file.path(outpath, paste0(LASfile_base, "_dtm_mosaic.tif")), overwrite = TRUE)
  
  dtm_smooth <- focal(dtm_mosaic, w = matrix(1, 25, 25), fun = mean, na.rm = TRUE, pad = TRUE)
  writeRaster(dtm_smooth, file.path(outpath, paste0(LASfile_base, "_dtm_mosaic_smooth.tif")), overwrite = TRUE)
  
  if (make_plot) plot(dtm_mosaic, bg = "white")
  dtm_prod <- terrain(dtm_mosaic, v = c("slope", "aspect"), unit = "radians")
  dtm_hillshade <- shade(dtm_prod$slope, dtm_prod$aspect)
  if (make_plot) plot(dtm_hillshade, col = gray(0:50/50), legend = FALSE)
  
  # Normalize point cloud
  opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_norm_tin")
  opt_filter(classified_ctg2) <- "-drop_as_withheld"
  ctg_norm_tin <- normalize_height(classified_ctg2, tin())
  if (make_plot) plot(ctg_norm_tin)
  
  # CHM generation
  opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_chm")
  rasterize_canopy(classified_ctg2, res = 0.1, algorithm = p2r(na.fill = knnidw()))
  chm_tiles <- list.files(path = outpath, pattern = '_chm.tif$', full.names = TRUE)
  chm_mosaic <- vrt(chm_tiles, overwrite = TRUE)
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
  
  chm_smooth <- focal(chm_filled, w = fgauss(2, 7))
  names(chm_smooth) <- 'Z'
  
  if (make_plot) {
    plot(chm_mosaic, col = height.colors(50))
    plot(chm_filled, col = height.colors(50))
    plot(chm_smooth, col = height.colors(50))
  }
  
  writeRaster(chm_mosaic, file.path(outpath, paste0(LASfile_base, "_chm_mosaic.tif")), overwrite = TRUE)
  writeRaster(chm_filled, file.path(outpath, paste0(LASfile_base, "_chm_filled.tif")), overwrite = TRUE)
  writeRaster(chm_smooth, file.path(outpath, paste0(LASfile_base, "_chm_smooth.tif")), overwrite = TRUE)
  
  # DSM generation
  opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_dsm")
  rasterize_canopy(classified_ctg2, res = 0.1, algorithm = p2r(na.fill = knnidw()))
  dsm_tiles <- list.files(path = outpath, pattern = '_dsm.tif$', full.names = TRUE)
  dsm_mosaic <- vrt(dsm_tiles, overwrite = TRUE)
  names(dsm_mosaic) <- 'Z'
  writeRaster(dsm_mosaic, file.path(outpath, paste0(LASfile_base, "_dsm_mosaic.tif")), overwrite = TRUE)
  
  dsm_filled <- focal(dsm_mosaic, w = 3, fun = "mean", na.policy = "only", na.rm = TRUE)
  dsm_smooth <- focal(dsm_filled, w = fgauss(1, 5))
  names(dsm_smooth) <- 'Z'
  
  writeRaster(dsm_filled, file.path(outpath, paste0(LASfile_base, "_dsm_filled.tif")), overwrite = TRUE)
  writeRaster(dsm_smooth, file.path(outpath, paste0(LASfile_base, "_dsm_smooth.tif")), overwrite = TRUE)
  
  if (make_plot) {
    plot(dsm_mosaic, col = height.colors(50))
    plot(dsm_filled, col = height.colors(50))
    plot(dsm_smooth, col = height.colors(50))
  }

  
  ######SuperSlow##################
  ttops <- locate_trees(ctg_norm_tin, lmf(4, hmin = 8.2, shape = "circular"))
  ttops
  
  library(sf)
  library(tibble)
  
  ttops_tiles <- list.files(path = outpath, pattern = '*shp', full.names = T)
  ttops_list <- lapply(ttops_tiles, read_sf)
  all_ttops <- do.call(rbind, ttops_list)
  all_ttops
  
  num_ttops<-seq(from=1,to=nrow(all_ttops))
  all_ttops$treeID<-num_ttops
  all_ttops
  if (make_plot){
  plot(all_ttops)
    
  }
  
  opt_output_files(ctg_norm_tin) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_segmented")
  algo <- dalponte2016(chm_mosaic, all_ttops, max_cr = 20, th_cr = 0.8, th_tree = 1)
  ctg_segmented <- segment_trees(ctg_norm_tin, algo)
  
  ctg_final = readLAScatalog(outpath, pattern = '_segmented.las')
  las = readLAS(ctg_final)
  
  plot(las, bg = "white", size = 4, color = "treeID")
    
  
  end_time = Sys.time()
  
  sink(output_file, append = TRUE)
  cat("Processing file:", LASfile_name, "\n")
  elapsed_seconds <- as.numeric(end_time - start_time, units = "secs")
  cat("Elapsed time (seconds):", elapsed_seconds, "\n\n")
  sink()
  
  
  # Remove the unwanted files
  all_files <- list.files(outpath, full.names = TRUE)
  file_keep_pattern <- "_summary.txt|_segmented.las"
  files_to_remove <- all_files[!grepl(file_keep_pattern, basename(all_files))]
  file.remove(files_to_remove)
  
  
  # --- Clear memory at end of iteration ---
  rm(list = setdiff(ls(), c("make_plot", "LASfile_dir", "las_files", "LASfile_path", "output_file")))
  gc()
  

  

