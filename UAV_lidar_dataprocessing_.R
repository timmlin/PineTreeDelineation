rm(list=ls(all=TRUE))
#install.packages("lidR")
library(lidR)
library(future)
library(RCSF)
library(terra)
library(lidRmetrics)

outpath = "G:/test/" 
setwd("G:/test")
#LASfile="G:/SLMACC/01_Terra/LiDAR/20221110_FlemingNMartin/20221110_FlemingNMartin_01/lidars/terra_las/cloud7072babcded3e1c7.las"
LASfile="G:/cloud14d82a2329c16cbe.las"

#do not run the following part if want to get quick result
las <- readLAS(LASfile)
print(las)
summary(las)
las_check(las)
las<- filter_duplicates(las)
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

# ---- Metrics -----

opt_select(ctg_norm_tin) <- "xyz"
opt_filter(ctg_norm_tin) <- "-drop_withheld -drop_z_below 0"
basic <- pixel_metrics(ctg_norm_tin , ~lidRmetrics::metrics_basic(Z), res = 1)
basic_tiles <- list.files(path = outpath, pattern = '_norm_tin.tif$', full.names = T)
basic_mosaic <- vrt(basic_tiles, overwrite = T)
writeRaster(basic_mosaic, filename = 'basic_metrics.tif', overwrite = TRUE)

# ----- Canopy Height Model -----
opt_output_files(ctg_norm_tin) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_chm")
#rasterize_canopy(ctg_norm_tin, res = 0.1, pitfree(c(0,2,5,10,15), c(0, 1.5)))

rasterize_canopy(ctg_norm_tin, res = 0.1,algorithm = p2r(na.fill = knnidw()))


chm_tiles <- list.files(path = outpath, pattern = '_chm.tif$', full.names = T)
chm_mosaic <- vrt(chm_tiles, overwrite = T)

#--- Remove extreme CHM values. Extreme points get 0 or 30 ---
chm_mosaic <- clamp(chm_mosaic, 0, 30, values = TRUE)
#--- Set layer name to Z ---
names(chm_mosaic) <- 'Z'

# ----- Fill CHM -----
chm_filled <- focal(chm_mosaic,w = 3,fun = "mean",na.policy = "only",na.rm = TRUE)
names(chm_filled) <- 'Z'
# ----- Smooth CHM -----
fgauss <- function(sigma, n = ws) {
  m <- matrix(ncol = n, nrow = n)
  col <- rep(1:n, n)
  row <- rep(1:n, each = n)
  x <- col - ceiling(n/2)
  y <- row - ceiling(n/2)
  m[cbind(row, col)] <- 1/(2 * pi * sigma^2) * exp(-(x^2 + y^2)/(2 * sigma^2))
  m/sum(m)
}

chm_smooth <- terra::focal(chm_filled,w = fgauss(1, n = 5))
names(chm_smooth) <- 'Z'

chm_mosaic
plot(chm_mosaic, col = height.colors(50))
plot(chm_filled, col = height.colors(50))
plot(chm_smooth, col = height.colors(50))

writeRaster(chm_mosaic, filename = 'chm_mosaic.tif', overwrite = T)
writeRaster(chm_filled, filename = 'chm_filled.tif', overwrite = T)
writeRaster(chm_smooth, filename = 'chm_smooth.tif', overwrite = T)

# ----- Digital Surface Model -----
opt_output_files(classified_ctg2) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_dsm")
rasterize_canopy(classified_ctg2,res = 0.1,algorithm = p2r(na.fill = knnidw()))
dsm_tiles <- list.files(path = outpath, pattern = '_dsm.tif$', full.names = T)
dsm_mosaic <- vrt(dsm_tiles, overwrite = T)
names(dsm_mosaic) <- 'Z'
writeRaster(dsm_mosaic, filename = 'dsm_mosaic.tif', overwrite = T)

# ----- Fill dsm -----
dsm_filled <- terra::focal(dsm_mosaic,w = 3,fun = "mean",na.policy = "only",na.rm = TRUE)
names(dsm_filled) <- 'Z'

# ----- Smooth dsm -----
dsm_smooth <- terra::focal(dsm_filled, w = fgauss(1, n = 5))
names(dsm_smooth) <- 'Z'

writeRaster(dsm_filled, filename = 'dsm_filled.tif', overwrite = T)
writeRaster(dsm_smooth, filename = 'dsm_smooth.tif', overwrite = T)


plot(dsm_mosaic, col = height.colors(50))
plot(dsm_filled, col = height.colors(50))
plot(dsm_smooth, col = height.colors(50))


######SuperSlow##################
ttops <- locate_trees(ctg_norm_tin, lmf(4))
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
plot(all_ttops)

opt_output_files(ctg_norm_tin) <- paste0(outpath, "/{XLEFT}_{YBOTTOM}_segmented")
algo <- dalponte2016(chm_mosaic, all_ttops)
ctg_segmented <- segment_trees(ctg_norm_tin, algo)

ctg_final = readLAScatalog(outpath, pattern = '_segmented')
las = readLAS(ctg_final)
plot(las, bg = "white", size = 4, color = "treeID")
