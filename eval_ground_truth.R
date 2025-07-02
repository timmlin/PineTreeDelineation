# Clear workspace and set working directory
rm(list = ls(all = TRUE))
setwd("/local/tli89/PineTreeDelineation")

# Load required libraries
library(lidR)
library(dplyr)
library(sf)
library(RANN)
library(tibble)
library(ggplot2)

# Initialize results list
results_list <- list()

# Iterate over tiles 1 to 112
for (tile_number in 1:112) {
  cat("Processing tile", tile_number, "\n")
  
  # Load segmented LAS files
  las_file_dir <- paste0("data/results/li/mohaka_segmented_las")
  las_files <- list.files(las_file_dir, pattern = paste0("tile_",tile_number,"_normalised_segmented"), full.names = TRUE)
  
  if (length(las_files) == 0) {
    warning(paste("No LAS files found for tile", tile_number))
    next
  }
  
  las_list <- lapply(las_files, readLAS)
  las_combined <- do.call(rbind, las_list)
  rm(las_list)
  
  if (is.null(las_combined)) {
    warning(paste("Invalid LAS data in tile", tile_number))
    next
  }
  
  # Extract highest point for each treeID
  highest_points <- las_combined@data %>%
    filter(!is.na(treeID)) %>%
    group_by(treeID) %>%
    slice_max(order_by = Z, n = 1, with_ties = FALSE) %>%
    ungroup()
  
  if (nrow(highest_points) == 0) {
    warning(paste("No valid treeIDs in tile", tile_number))
    next
  }
  
  las_crs <- projection(las_combined)
  
  if (is.na(las_crs) || las_crs == "") {
    las_crs <- st_crs(32617)  # or assign using EPSG directly
  } else {
    las_crs <- st_crs(las_crs)
  }
  
  
  predicted_trees_sf <- st_as_sf(highest_points, coords = c("X", "Y"), crs = las_crs)
  
  # Load ground truth
  shp_file_dir <- paste0("data/results/ground_truth/tile_", tile_number, "_ground_truth")
  shp_filename <- paste0("tile_", tile_number, ".shp")
  shp_path <- file.path(shp_file_dir, shp_filename)
  
  if (!file.exists(shp_path)) {
    warning(paste("Ground truth shapefile not found for tile", tile_number))
    next
  }
  
  ground_truth <- st_read(shp_path, quiet = TRUE)
  

  
  # Align CRS
  if (st_crs(predicted_trees_sf) != st_crs(ground_truth)) {
    ground_truth <- st_transform(ground_truth, st_crs(predicted_trees_sf))
  }
  
  # Compute nearest neighbors
  pred_coords <- st_coordinates(predicted_trees_sf)
  gt_coords <- st_coordinates(ground_truth)
  
  if (nrow(gt_coords) == 0) {
    warning(paste("No ground truth points in tile", tile_number))
    next
  }
  
  # Normalize coordinates to [0,1] range
  normalize_coords <- function(coords) {
    apply(coords, 2, function(x) (x - min(x)) / (max(x) - min(x)))
  }
  
  norm_pred_coords <- normalize_coords(pred_coords)
  norm_gt_coords <- normalize_coords(gt_coords)
  
  nn_result <- nn2(gt_coords, pred_coords, k = 1)
  
  # Match threshold
  threshold <- 2.0
  predicted_trees_sf$gt_index <- nn_result$nn.idx[, 1]
  predicted_trees_sf$gt_distance <- nn_result$nn.dists[, 1]
  predicted_trees_sf$match <- predicted_trees_sf$gt_distance <= threshold
  
  # Metrics
  true_positives <- sum(predicted_trees_sf$match)
  false_positives <- nrow(predicted_trees_sf) - true_positives
  false_negatives <- nrow(ground_truth) - true_positives
  
  precision <- ifelse((true_positives + false_positives) > 0,
                      true_positives / (true_positives + false_positives), NA)
  recall <- ifelse((true_positives + false_negatives) > 0,
                   true_positives / (true_positives + false_negatives), NA)
  f1_score <- ifelse(!is.na(precision) && !is.na(recall) && (precision + recall) > 0,
                     2 * (precision * recall) / (precision + recall), NA)
  
  # Store results
  results_list[[length(results_list) + 1]] <- tibble(
    tile = tile_number,
    true_positives = true_positives,
    false_positives = false_positives,
    false_negatives = false_negatives,
    precision = precision,
    recall = recall,
    f1_score = f1_score
  )
}

# Combine all results and write to CSV
results_df <- bind_rows(results_list)
write.csv(results_df, "li_mohaka_evaluation.csv", row.names = FALSE)

# Prepare data for plotting
predicted_trees_sf$match_label <- ifelse(predicted_trees_sf$match, "Matched", "Unmatched")
ground_truth$match_label <- "Ground Truth"

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



# Plot only ground truth
ggplot(ground_truth) +
  geom_sf(color = "grey30", shape = 4, size = 2, alpha = 0.7) +
  labs(title = "Ground Truth Tree Tops") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "none"
  )

# Plot only predicted tree top locations
ggplot(predicted_trees_sf) +
  geom_sf(color = "blue", shape = 16, size = 2, alpha = 0.7) +
  labs(title = "Predicted Tree Top Locations") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "none"
  )
