# Load necessary libraries
library(lidR)       # For working with LAS files
library(rgl)        # For 3D visualization

# Path to the uploaded LAS file
file_path <- "data/SCION/plot_31_annotated.las"

# Load the LAS file
las <- readLAS(file_path)

# Check if the file is loaded correctly
if (is.empty(las)) {
  stop("The LAS file could not be loaded. Please ensure the file path is correct.")
}

# Inspect the point cloud metadata
summary(las)

# Check the Classification field (if it exists)
if ("Classification" %in% names(las@data)) {
  cat("Classification field detected.\n")
} else {
  cat("No Classification field detected. The file might be unlabeled.\n")
}

# Visualize the point cloud colored by classification
if ("Classification" %in% names(las@data)) {
  plot(las, color = "Classification", legend = TRUE)
} else {
  # Default visualization if no classification is found
  plot(las, color = "Z", legend = TRUE)  # Coloring by elevation
}

# Optional: Save the visualization to a 3D viewer
# rgl.snapshot("segmentation_view.png")  # Save snapshot
