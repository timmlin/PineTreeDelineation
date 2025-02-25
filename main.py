import laspy
import numpy as np
import open3d as o3d
import random
from sklearn.cluster import MeanShift
from scipy.spatial import ConvexHull, Delaunay

from tools.utils import *

def in_hull(p, hull):
    """
    Check if a point is inside the convex hull.
    Args:
        p (ndarray): A 2D point (x, y).
        hull (ConvexHull): A scipy ConvexHull object.

    Returns:
        bool: True if the point is inside the convex hull, False otherwise.
    """
    delaunay = Delaunay(hull.points[hull.vertices])
    return delaunay.find_simplex(p) >= 0

def merge_clusters(new_clusters, all_clusters):
    """
    Merge clusters if a new cluster's centroid falls inside an existing cluster's convex hull.
    Clusters retain their assigned color across layers.

    Args:
        new_clusters (list of dict): Clusters detected in the current layer.
        all_clusters (list of dict): All accumulated clusters from previous layers.

    Returns:
        list: Updated list of `all_clusters`.
    """
    for new_cluster in new_clusters:
        centroid = new_cluster["centroid"][:2]  # (x, y) only

        for existing_cluster in all_clusters:
            prev_hull = ConvexHull(np.asarray(existing_cluster["convex_hull"].points)[:, :2])

            if in_hull(centroid, prev_hull):

                # Merge cluster points
                existing_cluster["points"] = np.vstack([existing_cluster["points"], new_cluster["points"]])
                existing_cluster["centroid"] = np.mean(existing_cluster["points"], axis=0)

                # Ensure color remains the same
                new_cluster["color"] = existing_cluster["color"]
                break
        else:
            # Assign a new unique color if no merge happened
            new_cluster["color"] = np.random.rand(3)
            all_clusters.append(new_cluster)

    return all_clusters

def layer_stacking(points, layer_height=1, visualise=True):
    tree_points, ground_points = points

    layers = []
    to_visualize = []

    min_height = np.min(tree_points[:, 2])
    max_height = np.max(tree_points[:, 2])
    num_layers = int(np.ceil((max_height - min_height) / layer_height))

    # ----------------------LAYERING-------------
    for i in reversed(range(num_layers)):
        cur_layer_min_height = i * layer_height
        cur_layer_max_height = cur_layer_min_height + layer_height

        layer_mask = (tree_points[:, 2] >= cur_layer_min_height) & (tree_points[:, 2] < cur_layer_max_height)
        layer_points = tree_points[layer_mask]

        if visualise:
            layer_colour = [random.random(), random.random(), random.random()]
            layer_cloud = o3d.geometry.PointCloud()
            layer_cloud.points = o3d.utility.Vector3dVector(layer_points)
            layer_cloud.colors = o3d.utility.Vector3dVector([layer_colour] * len(layer_points))
            to_visualize.append(layer_cloud)

        layers.append(layer_points)

    layers.append(ground_points)

    if visualise:
        ground_color = [1, 0, 0]  # Red
        ground_pnt_cld = o3d.geometry.PointCloud()
        ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
        ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_color] * len(ground_points))
        to_visualize.append(ground_pnt_cld)
        o3d.visualization.draw_geometries(to_visualize, window_name="Layers")

    # ---------------------------SEGMENTATION-------------------------------------
    visuals = []
    all_clusters = []  # Stores all clusters across layers

    for layer_num in range(1, 3):
        new_clusters = []  # Store new clusters detected in this layer
        to_cluster = np.vstack(layers[layer_num])
        layer_height = np.mean(to_cluster[:, 2]) 

        cluster = MeanShift(bandwidth=2).fit(to_cluster)
        labels = cluster.labels_
        unique_labels = np.unique(labels)
        centroids = cluster.cluster_centers_

        # Visualizing centroids
        centroids_colour = [1, 0, 0]  # Red
        centroids_pnt_cld = o3d.geometry.PointCloud()
        centroids_pnt_cld.points = o3d.utility.Vector3dVector(centroids)
        centroids_pnt_cld.colors = o3d.utility.Vector3dVector([centroids_colour] * len(centroids))

        clustered_point_clouds = []
        layer_convex_hulls = []

        for label in unique_labels:
            cluster_points = to_cluster[labels == label]
            xy_points = cluster_points[:, :2]
            z_mean = np.mean(cluster_points[:, 2])

            if len(xy_points) >= 3:  # Convex Hull requires at least 3 points
                hull = ConvexHull(xy_points)
                hull_vertices_xy = xy_points[hull.vertices]

                # Create 3D convex hull
                hull_vertices = np.hstack([hull_vertices_xy, np.full((len(hull_vertices_xy), 1), z_mean)])
                hull_edges = [(i, (i + 1) % len(hull.vertices)) for i in range(len(hull.vertices))]

                convex_hull = o3d.geometry.LineSet()
                convex_hull.points = o3d.utility.Vector3dVector(hull_vertices)
                convex_hull.lines = o3d.utility.Vector2iVector(hull_edges)
                convex_hull.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(hull_edges))  # Green

                layer_convex_hulls.append(convex_hull)

                # Store cluster data
                new_clusters.append({
                    "centroid": np.mean(cluster_points, axis=0),
                    "points": cluster_points,
                    "convex_hull": convex_hull
                })

        print(f"Layer {layer_num} segmentation completed \n \
            at a height of {layer_height} metres \n \
            {len(new_clusters)} trees found on this layer")

        # Merge clusters across layers while keeping colors
        all_clusters = merge_clusters(new_clusters, all_clusters)

        # Visualizing clusters with assigned colors
        for cluster in all_clusters:
            cluster_pnt_cld = o3d.geometry.PointCloud()
            cluster_pnt_cld.points = o3d.utility.Vector3dVector(cluster["points"])
            cluster_pnt_cld.colors = o3d.utility.Vector3dVector([cluster["color"]] * len(cluster["points"]))
            clustered_point_clouds.append(cluster_pnt_cld)

        # Visualization
        visuals.append(centroids_pnt_cld)
        visuals.extend(clustered_point_clouds)
        visuals.extend(layer_convex_hulls)

    o3d.visualization.draw_geometries(visuals)




def main():

    las_file = "data/UAV_sample_data/plot_31_pointcloud.las"
    las = laspy.read(las_file)

    #-------------PRE-PROCESSING
    
    las = noramlise_las(las)

    points = np.vstack((las.x, las.y, las.z)).T

    # Convert to Open3D Point Cloud to remove noise
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)

    pnt_cld, _ = pnt_cld.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    points = np.asarray(pnt_cld.points)


    #---------------GROUND-CLASSIFICATION--------------
    # pnt_cld = ransac_classify_ground(pnt_cld, visualise= True)
    points = classify_ground(points, visualise= False)


    # #---------------SEGMENTATION-----------------------
    layer_stacking(points, visualise = False)

    #---------------OTHER---------------------------
    # view_raw_cloud(points)








    
main()