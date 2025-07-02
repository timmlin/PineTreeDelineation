# Clear workspace and set working directory
rm(list = ls(all = TRUE))
setwd("/local/tli89/PineTreeDelineation")

library(lidR)
library(dplyr)
library(sf)
library(RANN)
library(tibble)


tile_number <- 17
# Load segmented LAS files
las_file_dir <- paste0("data/results/dbscan_dalponte/segmented_las/tile_", tile_number, "_las")
las_files <- list.files(las_file_dir, pattern = "_segmented.las$", full.names = TRUE)
las_list <- lapply(las_files, readLAS)
las_combined <- do.call(rbind, las_list)
rm(las_list)




# Extract highest point for each treeID (treetop candidates)
highest_points <- las_combined@data %>%
  filter(!is.na(treeID)) %>%
  group_by(treeID) %>%
  slice_max(order_by = Z, n = 1, with_ties = FALSE) %>%
  ungroup()

# Convert to sf object using LAS CRS
las_crs <- st_crs(las_combined)  # Attempt to retrieve CRS from LAS
if (is.null(las_crs)) {
  # Manually assign if LAS doesn't have CRS (adjust to match your data)
  las_crs <- st_crs(32617)  # Example: UTM Zone 17N
}
predicted_trees_sf <- st_as_sf(highest_points, coords = c("X", "Y"), crs = las_crs)

# Load ground truth shapefile
shp_file_dir <- paste0("data/results/ground_truth/tile_", tile_number, "_ground_truth")
shp_filename <- paste0("tile_", tile_number, ".shp")
ground_truth <- st_read(file.path(shp_file_dir, shp_filename))

# Align CRS: transform ground truth to predicted CRS
if (st_crs(predicted_trees_sf) != st_crs(ground_truth)) {
  ground_truth <- st_transform(ground_truth, st_crs(predicted_trees_sf))
}
# Extract XY coordinates for nearest neighbor matching
pred_coords <- st_coordinates(predicted_trees_sf)
gt_coords <- st_coordinates(ground_truth)

# 1-Nearest Neighbor search from predictions to ground truth
nn_result <- nn2(gt_coords, pred_coords, k = 1)

# Set threshold for match (in meters)
threshold <- 2.0  # Adjust as needed based on expected localization error

# Assign matching info
predicted_trees_sf$gt_index <- nn_result$nn.idx[, 1]
predicted_trees_sf$gt_distance <- nn_result$nn.dists[, 1]
predicted_trees_sf$match <- predicted_trees_sf$gt_distance <= threshold

# Evaluation metrics
true_positives <- sum(predicted_trees_sf$match)
false_positives <- nrow(predicted_trees_sf) - true_positives
false_negatives <- nrow(ground_truth) - true_positives

precision <- true_positives / (true_positives + false_positives)
recall <- true_positives / (true_positives + false_negatives)
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print results
cat("Evaluation Metrics:\n")
cat("True Positives:", true_positives, "\n")
cat("False Positives:", false_positives, "\n")
cat("False Negatives:", false_negatives, "\n")
cat("Precision:", round(precision, 3), "\n")
cat("Recall:", round(recall, 3), "\n")
cat("F1 Score:", round(f1_score, 3), "\n")






# Load ggplot2
library(ggplot2)

# Add match status to predicted points for plotting
predicted_trees_sf$match_label <- ifelse(predicted_trees_sf$match, "Matched", "Unmatched")

# Add a label for ground truth
ground_truth$match_label <- "Ground Truth"

# Combine both datasets into one for unified legend control
combined_data <- rbind(
  data.frame(geometry = st_geometry(predicted_trees_sf), match_label = predicted_trees_sf$match_label, shape = 16),
  data.frame(geometry = st_geometry(ground_truth), match_label = ground_truth$match_label, shape = 4)
)
combined_data <- st_as_sf(combined_data)

# Plot with unified legend
ggplot(combined_data) +
  geom_sf(aes(color = match_label, shape = match_label), size = 2, alpha = 0.7) +
  scale_color_manual(
    name = "Prediction Status",
    values = c("Matched" = "green", "Unmatched" = "red", "Ground Truth" = "grey30")
  ) +
  scale_shape_manual(
    name = "Prediction Status",
    values = c("Matched" = 16, "Unmatched" = 16, "Ground Truth" = 4)
  ) +
  labs(title = "Tree Top Prediction vs Ground Truth") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11),
    legend.position = "bottom"
  ) 

