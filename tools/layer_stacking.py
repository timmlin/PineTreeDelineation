import laspy
import numpy as np
import open3d as o3d
import random

from sklearn.cluster import DBSCAN, KMeans

from tools.utils import save_as_las_file


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

    # Compute the number of standard layers (excluding the top 4-meter layer)
    num_layers = int(np.ceil((max_height - min_height - 4) / layer_height))

    # ----------------------LAYERING-------------
    # Top layer (4 meters thick)
    top_layer_min_height = num_layers * layer_height
    top_layer_max_height = max_height  # Extends to the maximum height

    top_layer_mask = (tree_points[:, 2] >= top_layer_min_height) & (tree_points[:, 2] <= top_layer_max_height)
    top_layer_points = tree_points[top_layer_mask]

    if view_layers:
        top_layer_colour = [random.random(), random.random(), random.random()]
        top_layer_cloud = o3d.geometry.PointCloud()
        top_layer_cloud.points = o3d.utility.Vector3dVector(top_layer_points)
        top_layer_cloud.colors = o3d.utility.Vector3dVector([top_layer_colour] * len(top_layer_points))
        visualise_layers.append(top_layer_cloud)

    layers.append(top_layer_points)

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
    
    #---------------------------SEGMENTATION-------------------------------------
    visualise_segments = []
    all_clusters = []  # merged clusters
    previous_centroids = None  # centroids for K-Means initialization
    
    for layer_num in range(num_layers - 1):
        
        if len(layers[layer_num]) == 0:
            continue
        
        current_clusters = []
        to_cluster = np.vstack(layers[layer_num])
        
        if len(to_cluster) < 3:
            continue

        if layer_num == 0:  # Tree top detection using DBSCAN
            to_cluster_xy = to_cluster[:, :2]
            cluster = DBSCAN(eps=0.4, min_samples=20, metric='euclidean').fit(to_cluster_xy)

        else:  # K-Means with previous centroids as initialization
            kmeans = KMeans(n_clusters=len(previous_centroids), init=np.array(previous_centroids), n_init=1)
            cluster = kmeans.fit(to_cluster)

        labels = cluster.labels_
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Remove noise/unclustered points

        current_centroids = []
        for label in unique_labels:
            cluster_points = to_cluster[labels == label]
            
            if len(cluster_points) >= 3:
                centroid = np.mean(cluster_points, axis=0)
                current_clusters.append({
                    "points": cluster_points,
                    "centroid": centroid,
                })
                current_centroids.append(centroid)
        print(f"tree tops counted {len(current_clusters)}")
        if layer_num == 0:
            all_clusters = current_clusters
        else:
            all_clusters = merge_cluster_centroid(current_clusters, all_clusters)
        previous_centroids = current_centroids  # Update centroids for the next layer
        

    #---------------Visualisation-----------------------
    if view_clusters:
        visualise_segments = []

        for cluster in all_clusters:
            # Visualizing clusters
            cluster_colour = np.random.rand(3)
            cluster_pnt_cld = o3d.geometry.PointCloud()
            cluster_pnt_cld.points = o3d.utility.Vector3dVector(cluster["points"])
            cluster_pnt_cld.colors = o3d.utility.Vector3dVector([cluster_colour] * len(cluster["points"]))
            visualise_segments.append(cluster_pnt_cld)

        # # Centroid visualization as a sphere
        # centroid_color = [1, 0, 0]
        # centroid = cluster["centroid"]

        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.10) 
        # sphere.paint_uniform_color(centroid_color)  
        # sphere.translate(centroid +[0,0,2])  # Move sphere to centroid location

        # visualise_segments.append(sphere)




        ground_colour = [1, 0, 0]  # Red
        ground_pnt_cld = o3d.geometry.PointCloud()
        ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
        ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_colour] * len(ground_points))
        # visualise_segments.append(ground_pnt_cld)

    
        o3d.visualization.draw_geometries(visualise_segments, window_name="segmentation")
