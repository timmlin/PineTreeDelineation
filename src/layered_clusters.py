import laspy
import numpy as np
import open3d as o3d
import random
import os
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


                
def layered_clusters(points, layer_height=1, view_layers=False, view_clusters=True):
    tree_points, ground_points = points
    layers = []
    visualise_layers = []

    min_height = np.min(tree_points[:, 2])
    max_height = np.max(tree_points[:, 2])

    # Compute the number of standard layers, excluding the top 4-meter layer
    num_layers = int(np.ceil((max_height - min_height - 4) / layer_height))

    # ---------------------- Layering ----------------------

    #layer mask to apply to points to get top layer
    top_layer_min_height = num_layers * layer_height
    top_layer_mask = (tree_points[:,2] >= top_layer_min_height) & (tree_points[:, 2] < max_height)

    #applying layer mask    
    top_layer_points = tree_points[top_layer_mask]
    layers.append(top_layer_points)

    if view_layers: 
        visualize_layer(top_layer_points, visualise_layers, [1,0,0])

    # Create standard layers (excluding the top layer)

    for i in reversed(range(num_layers)):
        cur_layer_min_height = i * layer_height
        cur_layer_max_height = cur_layer_min_height + layer_height

        #layer mask to apply to points to get current layer
        layer_mask = (tree_points[:,2] >= cur_layer_min_height) & (tree_points[:, 2] < cur_layer_max_height)
        layer_points = tree_points[layer_mask]
        layers.append(layer_points)

        if view_layers and i > num_layers // 2:
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
        visualize_clusters(all_clusters, ground_points)

    
    return (all_clusters, ground_points)






def visualize_layer(points, visualise_layers, color = None):

    if color == None:
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

def visualize_clusters(all_clusters, ground_points):
    visualise_segments = []

    ground_cloud = o3d.geometry.PointCloud()
    ground_cloud.points = o3d.utility.Vector3dVector(ground_points)
    # visualise_segments.append(ground_cloud)

    
    for cluster in all_clusters:
        color = np.random.rand(3)
        cluster_cloud = o3d.geometry.PointCloud()
        cluster_cloud.points = o3d.utility.Vector3dVector(cluster["points"])
        cluster_cloud.colors = o3d.utility.Vector3dVector([color] * len(cluster["points"]))
        visualise_segments.append(cluster_cloud)
    # visualise_segments = visualise_segments[100:103]
    # visualise_segments.append(ground_cloud)
    o3d.visualization.draw_geometries(visualise_segments, window_name="Segmentation")




def save_classified_las(point_groups, ground_points, filename, base_header=None):
    """
    Save a LAS file with an extra byte dimension 'tree_id' for tree segmentation labels.

    Parameters:
    - point_groups: List of dicts with 'points' key (N, 3) arrays for each cluster/tree
    - ground_points: numpy array of (N, 3) ground points
    - filename: output .las or .laz file
    - base_header: optional laspy header
    """
    # Define header with point format 6+ (supports extra dimensions)
    if base_header is None:
        header = laspy.LasHeader(point_format=6, version="1.4")
    else:
        header = base_header

    # Define new extra byte dimension 'tree_id'
    if 'treeID' not in header.point_format.extra_dimensions:
        header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint32))

    las = laspy.LasData(header)

    all_points = []
    tree_ids = []

    for tree_id, group in enumerate(point_groups, start=1):
        pts = group["points"][:, :3]
        all_points.append(pts)
        tree_ids.append(np.full(len(pts), tree_id, dtype=np.uint32))

    # Add ground points, assign tree_id 0 or a reserved value
    all_points.append(ground_points[:, :3])
    tree_ids.append(np.full(len(ground_points), 0, dtype=np.uint32))

    merged_points = np.vstack(all_points)
    merged_tree_ids = np.hstack(tree_ids)

    las.x = merged_points[:, 0]
    las.y = merged_points[:, 1]
    las.z = merged_points[:, 2]
    las['treeID'] = merged_tree_ids  # Custom attribute

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    las.write(filename)
    print(f"Saved LAS with tree_id extra byte (max ID: {merged_tree_ids.max()}) to {filename}")

