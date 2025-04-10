rm(list=ls(all=TRUE))
library(tools) # Load tools package for file path manipulation
library(lidR)
library(future)
library(RCSF)
library(terra)
library(tidyverse)

# The directory path of the LAS file
LASfile_dir <- "data/SCION/UAV_lidar/grid"

# Define name of the LAS file
LASfile_name <- "tile_5_11.laz"


# Remove file extension if present
LASfile_base <- file_path_sans_ext(LASfile_name)

# Define output directory
outpath <- file.path(paste0("outputs/", LASfile_base, "_outputs"))

# Create the output directory if it doesn't exist
dir.create(outpath, recursive = TRUE, showWarnings = FALSE)

# Create the output file path
output_file <- file.path(outpath, paste0(LASfile_base, "_summary.txt"))


las <- readLAS(file.path(LASfile_dir, LASfile_name))
las<- filter_duplicates(las)
las_check(las)

#######################################################

ctg = readLAScatalog(file.path(LASfile_dir, LASfile_name))

plan(multisession, workers=3L)

# Create a new set of 250 x 250 m.las files
# but extended with a 10 m buffer in the folder

opt_chunk_buffer(ctg) <- 10
opt_chunk_size(ctg) <- 250
#opt_chunk_buffer(ctg) <- 10
#opt_chunk_size(ctg) <- 250
opt_filter(ctg) <- ""


opt_output_files(ctg) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_tiled")
newctg <- catalog_retile(ctg)

# ---- Classify ground points ----
opt_output_files(newctg) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_classifyground")
classified_ctg1 <- classify_ground(newctg, csf(class_threshold = 0.25, cloth_resolution = 0.25, rigidness = 2)) #parameter setting from UBC

# ---- Classify noise points ----
opt_output_files(classified_ctg1) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_classifynoise")
classified_ctg2<- classify_noise(classified_ctg1, ivf(5,6)) #Lastools same

# ----- DTM -----
opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_dtm")
rasterize_terrain(classified_ctg2, res = 0.1, tin(), overwrite=TRUE)

dtm_tiles <- list.files(path = outpath, pattern = '_dtm.tif$', full.names = T)
dtm_mosaic <- vrt(dtm_tiles, overwrite = T)
writeRaster(dtm_mosaic, filename = file.path(outpath, paste0(LASfile_base, "_dtm_mosaic.tif")), overwrite = TRUE)

# ----- DTM smooth-----
dtm_smooth <- dtm_mosaic %>%focal(w = matrix(1, 25, 25), fun = mean, na.rm = TRUE,pad = TRUE)
writeRaster(dtm_smooth, filename = file.path(outpath, paste0(LASfile_base, '_dtm_mosaic_smooth.tif')), overwrite = TRUE)

dtm_prod <- terra::terrain(dtm_mosaic, v = c("slope", "aspect"), unit = "radians")
dtm_hillshade <- terra::shade(slope = dtm_prod$slope, aspect = dtm_prod$aspect)

remove(dtm_mosaic)# Or the variable will eat too much memory and the following process will return an error
remove(dtm_smooth)# Or the variable will eat too much memory and the following process will return an error
# ---- Normalize point cloud ----

#opt_output_files(classified_ctg2) <-  paste0(outpath, "/{XLEFT}_{YBOTTOM}_norm")
#ctg_norm <- normalize_height(classified_ctg2, dtm_mosaic)

opt_output_files(classified_ctg2) <-  paste0(outpath, "/{XLEFT}_{YBOTTOM}_norm_tin")
opt_filter(classified_ctg2) <- '-drop_as_witheld'
ctg_norm_tin <- normalize_height(classified_ctg2, tin())

# List all the normalized chunk files with the suffix _norm_tin.las
norm_chunks <- list.files(path = outpath, pattern = '_norm_tin.las$', full.names = TRUE)

# Read and combine the chunks into one large point cloud
las_list <- lapply(norm_chunks, readLAS)

# Merge the point cloud chunks into one larger LAS object
las_combined <- do.call(rbind, las_list)

# Optionally, you can check the size of the combined point cloud
print(paste("Number of points in combined point cloud:", npoints(las_combined)))

# Save the reconstructed point cloud
output_combined_file <- file.path(outpath, paste0(LASfile_base, "_norm_combined.las"))
writeLAS(las_combined, output_combined_file)

# Print a success message
print(paste("Reconstructed normalized point cloud saved to:", output_combined_file))


