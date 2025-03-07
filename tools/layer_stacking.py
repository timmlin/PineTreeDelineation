import laspy
import numpy as np
import open3d as o3d
import random

from sklearn import datasets, linear_model
from sklearn.cluster import DBSCAN, KMeans, estimate_bandwidth
from scipy.spatial import ConvexHull



def hull_overlap(hull1, hull2):
    """
    Computes the overlap ratio between two convex hulls.

    Args:
        hull1, hull2 (ConvexHull): Convex hulls of two clusters.

    Returns:
        float: Overlap ratio (0 to 1).
    """
    points1 = hull1.points[hull1.vertices]
    points2 = hull2.points[hull2.vertices]

    # Combine all points and create a new convex hull
    combined_points = np.vstack([points1, points2])
    combined_hull = ConvexHull(combined_points)

    # If combined hull area is smaller than sum of both, they overlap
    intersection_area = (hull1.volume + hull2.volume) - combined_hull.volume
    smaller_hull_area = min(hull1.volume, hull2.volume)

    return max(0, intersection_area / smaller_hull_area)



def merge_cluster_convex_hull(cur_cluster, all_clusters, overlap_threshold):
    merged = False
    new_clusters = []

    for existing_cluster in all_clusters:
        overlap_ratio = hull_overlap(existing_cluster["convex_hull"], cur_cluster["convex_hull"])
        if overlap_ratio >= overlap_threshold:

            
            # Merge the clusters
            merged_points = np.vstack([existing_cluster["points"], cur_cluster["points"]])
            
            xy_points = merged_points[:, :2]
            xy_points = np.unique(xy_points, axis=0)  # Remove duplicates in XY plane
            
            merged_convex_hull = ConvexHull(xy_points)
            merged_centroid = np.mean(merged_points, axis=0)

            new_clusters.append({"points": merged_points,
                                "centroid": merged_centroid,
                                "convex_hull": merged_convex_hull})
            merged = True
        else:
            # Keep the existing cluster if it does not merge
            new_clusters.append(existing_cluster)

    if not merged:
        new_clusters.append(cur_cluster)  # If no merge, keep the cluster as is

    return new_clusters




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
            
            merged_convex_hull = ConvexHull(xy_points)
            merged_centroid = np.mean(merged_points, axis=0)

            new_clusters.append({"points": merged_points,
                                "centroid": merged_centroid,
                                "convex_hull": merged_convex_hull})
            merged = True
        else:
            # Keep the existing cluster if it does not merge
            new_clusters.append(existing_cluster)

    if not merged:
        new_clusters.append(cur_cluster)  # If no merge, keep the cluster as is

    return new_clusters


                
def layer_stacking(points, layer_height=1, view_layers=False, view_clusters=True):
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
    all_hulls = []

    for layer_num in range(num_layers -1 ):

        new_clusters = []

        to_cluster = np.vstack(layers[layer_num])

        if len(to_cluster) < 3:  # Skip clustering for too few points
            continue

        # Adjust DBSCAN parameters based on layer number
        if layer_num < 3:
            cluster = DBSCAN(eps=0.4, min_samples=int(np.log(len(to_cluster))), metric='euclidean').fit(to_cluster)
        elif layer_num < 5:
            cluster = DBSCAN(eps=0.3, min_samples=int(np.log(len(to_cluster))), metric='euclidean').fit(to_cluster)
        elif layer_num < 8:
            cluster = DBSCAN(eps=0.15, min_samples=int(np.log(len(to_cluster))) + 10, metric='euclidean').fit(to_cluster)
        else:
            cluster = DBSCAN(eps=0.3, min_samples=int(np.log(len(to_cluster))) + 10, metric='euclidean').fit(to_cluster)

        labels = cluster.labels_
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Removes unclustered points

        for label in unique_labels:
            cluster_points = to_cluster[labels == label]
            xy_points = cluster_points[:, :2]
            xy_points = np.unique(xy_points, axis=0)  # Remove duplicates in XY plane

            if len(xy_points) >= 3:
                convex_hull = ConvexHull(xy_points)
                hull_vertices_xy = xy_points[convex_hull.vertices]

                mean_z = np.mean(cluster_points[:, 2])

                # Convert 2D hull to 3D by adding mean Z value for visualization
                hull_vertices_3d = np.column_stack((hull_vertices_xy, np.full(len(hull_vertices_xy), mean_z)))
                all_hulls.append(hull_vertices_3d)

                new_cluster = {
                    "points": cluster_points,
                    "centroid": np.mean(cluster_points, axis=0),
                    "convex_hull": convex_hull,
                    "color": np.random.rand(3)
                }

    #-----------------MERGING---------------------------
                
                all_clusters = merge_cluster_centroid(new_cluster, all_clusters, 2)


        print(f"Layer {layer_num} segmentation completed \n \
    at a height of {mean_z} metres \n \
    {len(all_clusters)} trees found on this layer \n \
    {len(to_cluster)} total points")
    

    #--------------POST-PROCESSING----------------
    # final_clusters = []
    # for cluster in all_clusters:
    #     if len(cluster["points"]) >= 100:
    #         final_clusters.append(cluster)
    #---------------Visualisation-----------------------
    if view_clusters:
        visualise_segments = []
        visualise_convex_hulls = []

        for cluster in all_clusters:
            # Visualizing clusters
            cluster_colour = np.random.rand(3)
            cluster_pnt_cld = o3d.geometry.PointCloud()
            cluster_pnt_cld.points = o3d.utility.Vector3dVector(cluster["points"])
            cluster_pnt_cld.colors = o3d.utility.Vector3dVector([cluster_colour] * len(cluster["points"]))
            visualise_segments.append(cluster_pnt_cld)
        
        # Visualizing convex hull
        for hull_vertices in all_hulls:
            hull_edges = [(i, (i + 1) % len(hull_vertices)) for i in range(len(hull_vertices))]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(hull_vertices)
            line_set.lines = o3d.utility.Vector2iVector(hull_edges)
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(hull_edges))
            visualise_convex_hulls.append(line_set)

        # visualise_segments.extend(visualise_convex_hulls)

        ground_colour = [1, 0, 0]  # Red
        ground_pnt_cld = o3d.geometry.PointCloud()
        ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
        ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_colour] * len(ground_points))
        visualise_segments.append(ground_pnt_cld)

    
    o3d.visualization.draw_geometries(visualise_segments, window_name="segmentation")
