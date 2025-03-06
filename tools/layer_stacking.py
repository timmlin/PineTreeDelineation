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



def merge_clusters(current_layer_clusters, above_layer_clusters, overlap_threshold):
    """
    Merges clusters from the above layer into the current layer if they overlap,
    appending merged clusters to a new list.

    Args:
        current_layer_clusters (list): Clusters in the current layer.
        above_layer_clusters (list): Clusters in the layer above.
        overlap_threshold (float): Minimum overlap required to merge clusters.

    Returns:
        list: New list of clusters for the current layer after merging.
    """

    merged_clusters = []  # New list to store the merged clusters
    merged_above_clusters = set()  # Keep track of which clusters from above have been merged

    # Iterate over clusters in the current layer
    for cur_cluster in current_layer_clusters:
        cur_hull_vertices = cur_cluster["convex_hull"].points[cur_cluster["convex_hull"].vertices, :2]
        new_cluster = cur_cluster.copy()  # Create a new copy of the current cluster
        merged = False  # Flag to track if merging happens

        for above_cluster in above_layer_clusters:
            # Use the ID of the above_cluster for uniqueness
            above_cluster_id = id(above_cluster)

            if above_cluster_id in merged_above_clusters:
                continue  # Skip already merged clusters

            above_hull_vertices = above_cluster["convex_hull"].points[above_cluster["convex_hull"].vertices, :2]

            # Compute hull overlap
            overlap_ratio = hull_overlap(ConvexHull(cur_hull_vertices), ConvexHull(above_hull_vertices))

            if overlap_ratio >= overlap_threshold:
                # Merge the above-layer cluster into the current-layer cluster
                new_cluster["points"] = np.vstack([new_cluster["points"], above_cluster["points"]])
                new_cluster["centroid"] = np.mean(new_cluster["points"], axis=0)

                # Maintain color consistency
                above_cluster["color"] = new_cluster["color"]

                # Mark this cluster as merged using its ID
                merged_above_clusters.add(above_cluster_id)
                merged = True

        merged_clusters.append(new_cluster)  # Append the merged cluster to the new list

    # Add any unmerged clusters from the above layer to the new list
    for above_cluster in above_layer_clusters:
        if id(above_cluster) not in merged_above_clusters:
            merged_clusters.append(above_cluster)

    return merged_clusters  # Return the new list of merged clusters


def merge_clusters_centroids(current_layer_clusters, above_layer_clusters, distance_threshold):
    merged_clusters = []  # New list to store the merged clusters
    merged_clusters_set = set()  # Keep track of which clusters from above have been merged

    # Iterate over clusters in the above layer
    for prev_cluster in above_layer_clusters:
        prev_centroid = np.array(prev_cluster["centroid"][:2])
        new_cluster_points = prev_cluster["points"]  # Start with the points of the previous cluster

        # Flag to track if the current above layer cluster gets merged
        is_merged = False
        
        # Check each cluster in the current layer
        for cur_cluster in current_layer_clusters:
            cur_centroid = np.array(cur_cluster["centroid"][:2])
            
            # Calculate Euclidean distance between centroids
            distance = np.linalg.norm(cur_centroid - prev_centroid)
            
            if distance <= distance_threshold:
                # If clusters are close enough, merge them
                new_cluster_points = np.vstack([new_cluster_points, cur_cluster["points"]])
                is_merged = True  # Mark the cluster as merged
        
        # Create a merged cluster and append it
        merged_clusters.append({
            "points": new_cluster_points,
            "centroid": np.mean(new_cluster_points, axis=0)
        })

    # Add any unmerged clusters from the current layer (optional)
    for cur_cluster in current_layer_clusters:
        if not any(np.array_equal(cur_cluster["centroid"][:2], prev_cluster["centroid"][:2]) for prev_cluster in above_layer_clusters):
            merged_clusters.append({
                "points": cur_cluster["points"],
                "centroid": cur_cluster["centroid"]
            })

    return merged_clusters


                
def layer_stacking(points, layer_height=1, view_layers = False, view_clsuters = True):
    tree_points, ground_points = points

    layers = []
    visualise_layers = []

    min_height = np.min(tree_points[:, 2])
    max_height = np.max(tree_points[:, 2])
    num_layers = int(np.ceil((max_height - min_height) / layer_height))

    # ----------------------LAYERING-------------
    for i in reversed(range(num_layers) ):
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
    all_clusters = []  
    all_hulls = []

    for layer_num in range(num_layers -1 ):
        new_clusters = []  
        to_cluster = np.vstack(layers[layer_num])
        layer_height = np.mean(to_cluster[:, 2]) 

        if layer_num < 3:
            cluster = DBSCAN(eps=0.4, min_samples=int(np.log(len(to_cluster))), metric='euclidean').fit(to_cluster)
        elif layer_num < 5:
            cluster = DBSCAN(eps=0.3, min_samples= int(np.log(len(to_cluster))), metric='euclidean').fit(to_cluster)
        elif layer_num < 8:
            cluster = DBSCAN(eps=0.15, min_samples= int(np.log(len(to_cluster))) + 10, metric='euclidean').fit(to_cluster)
        else:
            cluster = DBSCAN(eps=0.3, min_samples= int(np.log(len(to_cluster))) + 10, metric='euclidean').fit(to_cluster)

        
        labels = cluster.labels_
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Removes unclustered points

        for label in unique_labels:
            cluster_points = to_cluster[labels == label]
            
            xy_points = cluster_points[:, :2]
            xy_points = np.unique(xy_points, axis = 0)

            if len(xy_points) >= 3:
                convex_hull = ConvexHull(xy_points)
                hull_vertices_xy = xy_points[convex_hull.vertices]

                mean_z = np.mean(cluster_points[:, 2])

                # Convert 2D hull to 3D by adding mean Z value for visulaisation
                hull_vertices_3d = np.column_stack((hull_vertices_xy, np.full(len(hull_vertices_xy), mean_z)))
                all_hulls.append(hull_vertices_3d)

                new_clusters.append({
                    "points": cluster_points,
                    "centroid": np.mean(cluster_points, axis=0),
                    "convex_hull": convex_hull,
                    "color": np.random.rand(3)
                })


        overlap_threshold = 0.005
        all_clusters = merge_clusters(new_clusters, all_clusters, overlap_threshold)
        # all_clusters = merge_clusters_centroids(new_clusters, all_clusters, 1)

        


        print(f"Layer {layer_num} segmentation completed \n \
    at a height of {layer_height} metres \n \
    {len(all_clusters)} trees found on this layer \n \
    {len(to_cluster)} total points")

    #---------------Visualisation-----------------------
        if view_clsuters:
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

            visualise_segments.extend(visualise_convex_hulls)

            ground_colour = [1, 0, 0]  # Red
            ground_pnt_cld = o3d.geometry.PointCloud()
            ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
            ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_colour] * len(ground_points))
            visualise_segments.append(ground_pnt_cld)

    
    o3d.visualization.draw_geometries(visualise_segments, window_name="segmentation")
