rm(list=ls(all=TRUE))
setwd("/local/tli89/PineTreeDelineation")

library(tools)
library(lidR)
library(future)
library(RCSF)
library(terra)
library(tidyverse)

outpath <- "segmented_output.las"

ctg_final = readLAScatalog(outpath, pattern = '_segmented.las')
las = readLAS(ctg_final)

# Exclude the last (largest) treeID
max_id <- min(las$treeID, na.rm = TRUE)
las_filtered <- filter_poi(las, treeID != max_id)

plot(las_filtered, bg = "white", size = 4, color = "treeID")
