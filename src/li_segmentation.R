rm(list = ls(globalenv()))
# Load packages
library(lidR)
library(sf)
library(terra)
library(sys)

# --- User Parameters ---

dir_to_check <- "path-to-output-directory"
LASfile_dir <- "path-to-input-las-file"

# li2012 algorithm parameters
li2012_R <- 4
li2012_speed_up <- 5

# --- Workflow ---

# Get list of LAS files
las_files <- list.files(LASfile_dir, pattern = "\\.las$", full.names = TRUE)

for (las_path in las_files) {
  las_name <- basename(las_path)
  cat("Processing:", las_name, "\n")
  
  # Define output (segmented) file path
  out_path <- file.path(dir_to_check, paste0(tools::file_path_sans_ext(las_name), "_segmented.las"))
  
  # Only process if the segmented file already exists
  if (file.exists(out_path)) {
    
    # Read original LAS file
    las <- readLAS(las_path)
    
    # Segment trees
    start.time <- Sys.time()
    las_segmented <- segment_trees(las, li2012(R = li2012_R, speed_up = li2012_speed_up))
    end.time <- Sys.time()
    
    # Count number of trees
    tree_count <- length(unique(las_segmented$treeID[!is.na(las_segmented$treeID)]))
    
    # Calculate time elapsed
    elapsed <- as.numeric(difftime(end.time, start.time, units = "secs"))
    
    # Only write the header once at the beginning
    if (!file.exists(log_file)) {
      write.table(log_data, file = log_file, sep = ",", row.names = FALSE, col.names = TRUE)
    }
    
    # Log processing information
    new_row <- data.frame(
      file = las_name,
      time_seconds = elapsed,
      tree_count = tree_count,
      stringsAsFactors = FALSE
    )
    
    write.table(new_row, file = log_file, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
  } else {
    cat("Segmented file does not exist, skipping:", las_name, "\n")
  }
}
