library("sf")
library("lidR")
library("raster")

grid_path <- "data/UTF-8Square_grids_50m/Square_grids_50m.shp"
trees_locs_path <- "data/SCION/tree_top_points_ground_truth/tree top points 7th dec.shp"
laz_path <- "data/SCION/UAV_lidar/L1_mohaka_173_lidar.laz"



# Read data
grid <- st_read(grid_path)
trees <- st_read(trees_locs_path)
ctg <- readLAScatalog(laz_path)

plot(grid)

# Create output directories
dir.create("data/SCION/tree_tiles", showWarnings = FALSE)
dir.create("data/SCION/las_tiles", showWarnings = FALSE)

# Ensure grid polygons have unique IDs
if (!"ID" %in% names(grid)) {
  grid$ID <- seq_len(nrow(grid))
}

# --- 1. Tile tree points using irregular grid polygons ---
library(sf)
library(dplyr)

# Ensure unique ID in grid
if (!"ID" %in% names(grid)) {
  grid$ID <- seq_len(nrow(grid))
}

# Drop unnecessary attributes from grid before spatial join
grid_simple <- grid %>% select(ID, geometry)

# Process each polygon
for (i in seq_len(nrow(grid_simple))) {
  tile <- grid_simple[i, ]
  
  # Spatial intersection (clip trees to this tile)
  trees_tile <- st_intersection(trees, tile)
  
  if (nrow(trees_tile) > 0) {
    # Output file base name
    out_base <- paste0("data/SCION/tree_tiles/tile_", tile$ID)
    
    # Clean up old files manually
    exts <- c(".shp", ".shx", ".dbf", ".prj", ".cpg")
    lapply(paste0(out_base, exts), unlink, force = TRUE)
    
    # Avoid duplicated column names
    trees_tile <- trees_tile %>% select(-matches("ID\\.1"))  # remove overlapping ID field if present
    
    # Write shapefile
    st_write(trees_tile, paste0(out_base, ".shp"), driver = "ESRI Shapefile")
  }
}

# --- 2. Clip the LAZ file using the irregular polygon grid ---

st_crs(grid)
st_crs(ctg)

grid <- st_transform(grid, crs = projection(ctg))


clip_tile <- function(las, tile)
{
  las <- clip_roi(las, tile)
  if (is.empty(las)) return(NULL)
  return(las)
}



# Add ID column if it doesn't exist
if (!"ID" %in% names(grid)) grid$ID <- seq_len(nrow(grid))

opt_output_files(ctg) <- "data/SCION/las_tiles/tile_{ID}"

catalog_apply(ctg, FUN = clip_tile, roi = grid)


