setwd("C:/Users/lindb/Desktop/PineTreeDelineation")
rm(list=ls(all=TRUE))
#install.packages("lidR")
library(lidR)
library(future)
library(RCSF)
library(terra)
library(lidRmetrics)

outpath = "data/SCION/summary" 

# Define the path to the LAS file
LASfile <- "data/SCION/L1_mohaka_173_lidar.laz"

# Extract the directory path of the LAS file
LASfile_dir <- dirname(LASfile)

# Extract file name without extension
LASfile_name <- tools::file_path_sans_ext(basename(LASfile))

# Create the output file name in the same directory as the LAS file
output_file <- file.path(LASfile_dir, paste0(LASfile_name, "_summary.txt"))
# Redirect output to the summary txt file
sink(output_file)


las <- readLAS(LASfile)
print(las)
summary(las)
las_check(las)
las<- filter_duplicates(las)

#stop redirecting output
sink()
#######################################################
ctg = readLAScatalog(LASfile)
plot(ctg)

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

plot(newctg,chunk=TRUE)

# ---- Classify ground points ----
opt_output_files(newctg) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_classifyground")
classified_ctg1 <- classify_ground(newctg, csf(class_threshold = 0.25, cloth_resolution = 0.25, rigidness = 2)) #parameter setting from UBC

# ---- Classify noise points ----
opt_output_files(classified_ctg1) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_classifynoise")
classified_ctg2<- classify_noise(classified_ctg1, ivf(5,6)) #Lastools same

plot(classified_ctg2,chunk=TRUE)

# ----- DTM -----
opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_dtm")
rasterize_terrain(classified_ctg2, res = 0.1, tin())


dtm_tiles <- list.files(path = outpath, pattern = '_dtm.tif$', full.names = T)
dtm_mosaic <- vrt(dtm_tiles, overwrite = T)
dtm_mosaic
writeRaster(dtm_mosaic, filename = 'dtm_mosaic.tif', overwrite = T)

# ----- DTM smooth-----
dtm_smooth <- dtm_mosaic %>%focal(w = matrix(1, 25, 25), fun = mean, na.rm = TRUE,pad = TRUE)
writeRaster(dtm_smooth, filename = 'dtm_mosaic_smooth.tif', overwrite = T)

plot(dtm_mosaic, bg = "white") 
dtm_prod <- terra::terrain(dtm_mosaic, v = c("slope", "aspect"), unit = "radians")
dtm_hillshade <- terra::shade(slope = dtm_prod$slope, aspect = dtm_prod$aspect)
plot(dtm_hillshade, col = gray(0:50/50), legend = FALSE)


remove(dtm_mosaic)# Or the variable will eat too much memory and the following process will return an error
remove(dtm_smooth)# Or the variable will eat too much memory and the following process will return an error
# ---- Normalize point cloud ----

#opt_output_files(classified_ctg2) <-  paste0(outpath, "/{XLEFT}_{YBOTTOM}_norm")
#ctg_norm <- normalize_height(classified_ctg2, dtm_mosaic)

opt_output_files(classified_ctg2) <-  paste0(outpath, "/{XLEFT}_{YBOTTOM}_norm_tin")
opt_filter(classified_ctg2) <- '-drop_as_witheld'
ctg_norm_tin <- normalize_height(classified_ctg2, tin())


#Set output file name for the normalized point cloud
output_norm_file <- file.path(outpath, paste0(LASfile_name, "_normalized.las"))

# Save the normalized point cloud to a new LAS file
writeLAS(ctg_norm_tin, output_norm_file)

# Confirm that the file has been saved
cat("Normalized point cloud saved to:", output_norm_file, "\n")
