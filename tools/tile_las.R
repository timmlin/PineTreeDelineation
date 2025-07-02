library(lidR)
library(sf)

# Read the shapefile (irregular tiling grid)
grid <- st_read("data/UTF-8Square_grids_50m/Square_grids_50m.shp")

# Ensure 'ID' exists for naming
if (!"ID" %in% names(grid)) grid$ID <- seq_len(nrow(grid))

# Read the LAScatalog
ctg <- readLAScatalog("data/SCION/UAV_lidar/L1_mohaka_173_lidar.laz")

# Ensure CRS are aligned
# if (is.na(st_crs(ctg))) projection(ctg) <- st_crs(grid)$wkt
# grid <- st_transform(grid, crs = projection(ctg))

# Prepare output folder
dir.create("data/SCION/las_tiles", showWarnings = FALSE)

# Set LAScatalog options
opt_laz_compression(ctg) <- TRUE
opt_chunk_buffer(ctg) <- 0
opt_chunk_size(ctg) <- 0  # Means: don't split data into grid tiles

# Process each polygon in a loop
for (i in seq_len(nrow(grid))) {
  polygon <- grid[i, ]
  tile_id <- polygon$ID
  
  message(sprintf("Processing tile %s (%d of %d)...", tile_id, i, nrow(grid)))
  
  las_tile <- clip_roi(ctg, polygon)
  
  if (!is.null(las_tile) && !is.empty(las_tile)) {
    writeLAS(las_tile, paste0("data/SCION/las_tiles/tile_", tile_id, ".laz"))
  } else {
    message(sprintf("No points found in tile %s", tile_id))
  }
}
