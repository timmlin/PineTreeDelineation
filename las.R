#!/usr/bin/env Rscript

# Load necessary libraries
suppressMessages({
  library(lidR)
  library(rgl) # For 3D visualization
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if a file path is provided
if (length(args) < 1) {
  cat("Usage: Rscript view_las.R <path_to_las_or_laz_file>\n")
  quit(status = 1)
}

# Get the file path from arguments
file_path <- args[1]

# Check if the file exists
if (!file.exists(file_path)) {
  cat("Error: File not found at", file_path, "\n")
  quit(status = 1)
}

# Load the LAS/LAZ file
cat("Loading LAS/LAZ file...\n")
las <- tryCatch(
  readLAS(file_path),
  error = function(e) {
    cat("Error: Unable to read the LAS/LAZ file. Ensure it is valid.\n")
    quit(status = 1)
  }
)

# Check if the file was loaded correctly
if (is.null(las)) {
  cat("Error: Invalid LAS/LAZ file.\n")
  quit(status = 1)
}

# Display basic information about the LAS file
cat("LAS File Information:\n")
print(las@header)

# Plot the LAS file
cat("Rendering 3D visualization...\n")
tryCatch({
  plot(las)
}, error = function(e) {
  cat("Error: Unable to render the plot. Ensure the 'rgl' package is installed.\n")
  quit(status = 1)
})
