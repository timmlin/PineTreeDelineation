# Load necessary libraries
library(lidR)       # For working with LAS files
library(rgl)        # For 3D visualization

# Path to the uploaded LAS file
path <- "data/rolleston_forest_plots/plot_22_outputs"

ctg_final = readLAScatalog(path, pattern = '_segmented.las')
las = readLAS(ctg_final)

summary(las)

# Identify available color attributes
available_attributes <- names(las@data)
plot(las, color = "Z", legend = TRUE)  # Coloring by elevation


# Optional: Save the visualization to a 3D viewer
# rgl.snapshot("pointcloud_view.png")  # Save snapshot
