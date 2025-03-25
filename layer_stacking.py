import laspy
import numpy as np
import open3d as o3d
import random

from sklearn.cluster import DBSCAN, KMeans



def merge_cluster_centroid(current_clusters, all_clusters):
    new_clusters = []

    # To keep track of clusters from all_clusters that are not merged
    merged_clusters = set()  # Will hold the indices of clusters that got merged

    for cur_cluster in current_clusters:
        cur_centroid = np.array(cur_cluster["centroid"])[:2]
        merged = False
        min_distance = float('inf')

        for i, existing_cluster in enumerate(all_clusters):
            if i in merged_clusters:  # Skip already merged clusters
                continue

            existing_centroid = np.array(existing_cluster["centroid"])[:2]
            distance = np.linalg.norm(cur_centroid - existing_centroid)

            if distance <= min_distance:  # check distance between centroids
                min_distance = distance
                closest_cluster = existing_cluster
                closest_index = i

        # Merge the clusters
        new_points = np.vstack((cur_cluster["points"], closest_cluster["points"]))
        new_centroid = np.mean(new_points, axis=0)

        # Add the merged cluster to the new clusters list
        new_clusters.append({'points': new_points, 'centroid': new_centroid})

        # Mark the merged cluster as merged
        merged_clusters.add(closest_index)

    # Add all the clusters from `all_clusters` that were not merged
    for i, existing_cluster in enumerate(all_clusters):
        if i not in merged_clusters:
            new_clusters.append(existing_cluster)

    return new_clusters


                
def layer_stacking(points, layer_height=1, view_layers=False, view_clusters=True):
    tree_points, ground_points = points
    layers = []
    visualise_layers = []

    min_height = np.min(tree_points[:, 2])
    max_height = np.max(tree_points[:, 2])

    # Compute the number of standard layers, excluding the top 4-meter layer
    num_layers = int(np.ceil((max_height - min_height - 4) / layer_height))

    # ---------------------- Layering ----------------------
    def create_layer_mask(points, min_h, max_h):
        return (points[:, 2] >= min_h) & (points[:, 2] < max_h)

    # Create the top layer (4 meters thick)
    top_layer_min_height = num_layers * layer_height
    top_layer_mask = create_layer_mask(tree_points, top_layer_min_height, max_height)
    top_layer_points = tree_points[top_layer_mask]
    layers.append(top_layer_points)

    if view_layers:
        visualize_layer(top_layer_points, visualise_layers)

    # Create standard layers (excluding the top layer)
    for i in reversed(range(num_layers)):
        cur_layer_min_height = i * layer_height
        cur_layer_max_height = cur_layer_min_height + layer_height
        
        layer_mask = create_layer_mask(tree_points, cur_layer_min_height, cur_layer_max_height)
        layer_points = tree_points[layer_mask]
        layers.append(layer_points)

        if view_layers:
            visualize_layer(layer_points, visualise_layers)
    
    if view_layers:
        visualize_ground(ground_points, visualise_layers)
        o3d.visualization.draw_geometries(visualise_layers, window_name="Layers")
    
    # ---------------------- Segmentation ----------------------
    all_clusters = []
    previous_centroids = None

    for layer_num in range(num_layers - 1):
        if len(layers[layer_num]) == 0:
            continue
        current_clusters = []
        to_cluster = np.vstack(layers[layer_num])

        if len(to_cluster) < 3:
            continue

        # Cluster the top layer using DBSCAN, others using K-Means
        if layer_num == 0:
            cluster = DBSCAN(eps=0.4, min_samples=20).fit(to_cluster[:, :2])
        else:
            cluster = KMeans(n_clusters=len(previous_centroids), init=np.array(previous_centroids), n_init=1).fit(to_cluster)
        
        labels = cluster.labels_
        unique_labels = unique_labels = np.unique(labels[labels != -1])  # Remove noise
        current_centroids = []

        for label in unique_labels:
            cluster_points = to_cluster[labels == label]
            if len(cluster_points) >= 3:
                centroid = np.mean(cluster_points, axis=0)
                current_clusters.append({"points": cluster_points, "centroid": centroid})
                current_centroids.append(centroid)
        

        if layer_num == 0:
            all_clusters = current_clusters  
        else:
            all_clusters = merge_cluster_centroid(current_clusters, all_clusters)
        
        previous_centroids = current_centroids
    
    # ---------------------- Visualization ----------------------
    if view_clusters:
        visualize_clusters(all_clusters)


def visualize_layer(points, visualise_layers):
    color = [random.random(), random.random(), random.random()]
    layer_cloud = o3d.geometry.PointCloud()
    layer_cloud.points = o3d.utility.Vector3dVector(points)
    layer_cloud.colors = o3d.utility.Vector3dVector([color] * len(points))
    visualise_layers.append(layer_cloud)

def visualize_ground(ground_points, visualise_layers):
    ground_cloud = o3d.geometry.PointCloud()
    ground_cloud.points = o3d.utility.Vector3dVector(ground_points)
    ground_cloud.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(ground_points))  # Red color
    visualise_layers.append(ground_cloud)

def visualize_clusters(all_clusters):
    visualise_segments = []
    for cluster in all_clusters:
        color = np.random.rand(3)
        cluster_cloud = o3d.geometry.PointCloud()
        cluster_cloud.points = o3d.utility.Vector3dVector(cluster["points"])
        cluster_cloud.colors = o3d.utility.Vector3dVector([color] * len(cluster["points"]))
        visualise_segments.append(cluster_cloud)
    o3d.visualization.draw_geometries(visualise_segments, window_name="Segmentation")