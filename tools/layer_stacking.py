import laspy
import numpy as np
import open3d as o3d
import random

from sklearn.cluster import DBSCAN

from tools.utils import save_as_las_file







def merge_cluster_centroid(cur_cluster, all_clusters, distance_threshold):
    
    merged = False
    new_clusters = []

    for existing_cluster in all_clusters:    
            
        cur_centroid = cur_cluster["centroid"][:2]
        existing_centroid = existing_cluster["centroid"][:2] 
        
        distance = np.linalg.norm(cur_centroid- existing_centroid)
        
        if distance <= distance_threshold:
            # Merge the clusters
            merged_points = np.vstack([existing_cluster["points"], cur_cluster["points"]])
            
            xy_points = merged_points[:, :2]
            xy_points = np.unique(xy_points, axis=0)  # Remove duplicates in XY plane
            
            
            merged_centroid = np.mean(merged_points, axis=0)

            new_clusters.append({"points": merged_points,
                                "centroid": merged_centroid})
            merged = True
        else:
            # Keep the existing cluster if it does not merge
            new_clusters.append(existing_cluster)

    if not merged:
        new_clusters.append(cur_cluster)  # If no merge, keep the cluster as is

    return new_clusters


                
def layer_stacking(points, layer_height=1, view_layers=False, view_clusters=True, save_file = False):
    tree_points, ground_points = points

    layers = []
    visualise_layers = []

    min_height = np.min(tree_points[:, 2])
    max_height = np.max(tree_points[:, 2])
    num_layers = int(np.ceil((max_height - min_height) / layer_height))

    # ----------------------LAYERING-------------
    for i in reversed(range(num_layers)):
        cur_layer_min_height = i * layer_height
        cur_layer_max_height = cur_layer_min_height + layer_height

        layer_mask = (tree_points[:, 2] >= cur_layer_min_height) & (tree_points[:, 2] < cur_layer_max_height)
        layer_points = tree_points[layer_mask]

        if view_layers:
            layer_colour = [random.random(), random.random(), random.random()]
            layer_cloud = o3d.geometry.PointCloud()
            layer_cloud.points = o3d.utility.Vector3dVector(layer_points)
            layer_cloud.colors = o3d.utility.Vector3dVector([layer_colour] * len(layer_points))
            visualise_layers.append(layer_cloud)

        layers.append(layer_points)

    if view_layers:
        ground_colour = [1, 0, 0]  # Red
        ground_pnt_cld = o3d.geometry.PointCloud()
        ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
        ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_colour] * len(ground_points))
        visualise_layers.append(ground_pnt_cld)
        o3d.visualization.draw_geometries(visualise_layers, window_name="Layers")

    # ---------------------------SEGMENTATION-------------------------------------
    visualise_segments = []
    all_clusters = []  # Holds merged clusters

    for layer_num in range(num_layers - 1):
        if  len(layers[layer_num]) == 0:  # Check if sublist is empty
            continue
        
        new_clusters = []

        to_cluster = np.vstack(layers[layer_num])

        if len(to_cluster) < 3:  # Skip clustering for too few points
            continue

        # Adjust DBSCAN parameters based on layer number
        if layer_num < 3:
            cluster = DBSCAN(eps=0.4, min_samples=int(np.log(len(to_cluster))), metric='euclidean').fit(to_cluster)
            distance_threshold =  2.5

        elif layer_num < 5:
            cluster = DBSCAN(eps=0.3, min_samples=int(np.log(len(to_cluster))), metric='euclidean').fit(to_cluster)
            distance_threshold =  3

        elif layer_num < 10:
            cluster = DBSCAN(eps=0.15, min_samples=int(np.log(len(to_cluster))) + 10, metric='euclidean').fit(to_cluster)
            distance_threshold =  4
            
        else:
            cluster = DBSCAN(eps=0.3, min_samples=int(np.log(len(to_cluster))) + 10, metric='euclidean').fit(to_cluster)
            distance_threshold =  2


        labels = cluster.labels_
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Removes unclustered points

        for label in unique_labels:
            cluster_points = to_cluster[labels == label]
            xy_points = cluster_points[:, :2]
            xy_points = np.unique(xy_points, axis=0)  # Remove duplicates in XY plane

            if len(xy_points) >= 3:

                mean_z = np.mean(cluster_points[:, 2])

                new_cluster = {
                    "points": cluster_points,
                    "centroid": np.mean(cluster_points, axis=0),
                    "color": np.random.rand(3)
                }

    #-----------------MERGING---------------------------
                
                all_clusters = merge_cluster_centroid(new_cluster, all_clusters, 2.5)


        print(f"Layer {layer_num} segmentation completed \n \
    at a height of {round(mean_z, 2)} metres \n \
    {len(to_cluster)} total points")
    
    mean_points_per_tree = np.mean([arr['points'].shape[0] for arr in all_clusters])
    print(f'mean number of points per tree: {mean_points_per_tree}')
    
    #--------------POST-PROCESSING----------------
    print(f"post processing started")
    large_clusters = [c for c in all_clusters if len(c["points"]) >= int(mean_points_per_tree*0.8)]
    small_clusters = [c for c in all_clusters if len(c["points"]) <  int(mean_points_per_tree*0.8)]
    
    for small in small_clusters:
        min_distance = float('inf')
        closest_large = None
        
        for large in large_clusters:
            dist = np.linalg.norm(small["centroid"][:2] - large["centroid"][:2])
            if dist < min_distance:
                min_distance = dist
                closest_large = large
        
        if closest_large is not None:
            closest_large["points"] = np.vstack([closest_large["points"], small["points"]])
            closest_large["centroid"] = np.mean(closest_large["points"], axis=0)
    
    final_clusters = large_clusters
    print(f"Total number of trees after merging: {len(final_clusters)}")
    print()

    #---------------Visualisation-----------------------
    if view_clusters:
        visualise_segments = []

        for cluster in final_clusters:
            # Visualizing clusters
            cluster_colour = np.random.rand(3)
            cluster_pnt_cld = o3d.geometry.PointCloud()
            cluster_pnt_cld.points = o3d.utility.Vector3dVector(cluster["points"])
            cluster_pnt_cld.colors = o3d.utility.Vector3dVector([cluster_colour] * len(cluster["points"]))
            visualise_segments.append(cluster_pnt_cld)
        

        ground_colour = [1, 0, 0]  # Red
        ground_pnt_cld = o3d.geometry.PointCloud()
        ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
        ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_colour] * len(ground_points))
        visualise_segments.append(ground_pnt_cld)

    
        o3d.visualization.draw_geometries(visualise_segments, window_name="segmentation")

