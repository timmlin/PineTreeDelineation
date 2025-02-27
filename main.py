import laspy
import numpy as np
import open3d as o3d
import random
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from scipy.spatial import ConvexHull

from tools.utils import *


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



def merge_clusters(new_clusters, all_clusters, overlap_threshold):
    for new_cluster in new_clusters:
        new_hull_vertices = new_cluster["convex_hull"].points[new_cluster["convex_hull"].vertices, :2]

        for existing_cluster in all_clusters:
            existing_hull_vertices = existing_cluster["convex_hull"].points[existing_cluster["convex_hull"].vertices, :2]

            # compute hull overlap
            overlap_ratio = hull_overlap(ConvexHull(existing_hull_vertices), ConvexHull(new_hull_vertices))

            if overlap_ratio >= overlap_threshold:

                # Merge cluster points
                existing_cluster["points"] = np.vstack([existing_cluster["points"], new_cluster["points"]])
                existing_cluster["centroid"] = np.mean(existing_cluster["points"], axis=0)

                # Maintain color consistency
                new_cluster["color"] = existing_cluster["color"]
                break
        else:
            # Assign a new color if no merge occurred
            new_cluster["color"] = np.random.rand(3)
            all_clusters.append(new_cluster)

    return all_clusters



def layer_stacking(points, layer_height=1, view_layers = False, view_clsuters = True):
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

        print(f"layer number {i} has {len(layer_points)} points ")
        layers.append(layer_points)


    if view_layers:
        ground_color = [1, 0, 0]  # Red
        ground_pnt_cld = o3d.geometry.PointCloud()
        ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
        ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_color] * len(ground_points))
        visualise_layers.append(ground_pnt_cld)
        o3d.visualization.draw_geometries(visualise_layers, window_name="Layers")

    # ---------------------------SEGMENTATION-------------------------------------
    visualise_segments = []
    all_clusters = []  
    all_hulls = []

    for layer_num in range(2):
        new_clusters = []  
        to_cluster = np.vstack(layers[layer_num])
        layer_height = np.mean(to_cluster[:, 2]) 

        # cluster = MeanShift(bandwidth=2).fit(to_cluster)
        cluster = DBSCAN(eps=0.3, min_samples=5, metric='euclidean').fit(to_cluster)

        labels = cluster.labels_
        unique_labels = np.unique(labels)


        for label in unique_labels:
            cluster_points = to_cluster[labels == label]
            
            xy_points = cluster_points[:, :2]
            xy_points = np.unique(xy_points, axis = 0)

            if len(xy_points) >= 3:
                convex_hull = ConvexHull(xy_points)
                hull_vertices_xy = xy_points[convex_hull.vertices]

                mean_z = np.mean(cluster_points[:, 2])

                # Convert 2D hull to 3D by adding mean Z value
                hull_vertices_3d = np.column_stack((hull_vertices_xy, np.full(len(hull_vertices_xy), mean_z)))
                all_hulls.append(hull_vertices_3d)

                new_clusters.append({
                    "centroid": np.mean(cluster_points, axis=0),
                    "points": cluster_points,
                    "convex_hull": convex_hull,
                    "hull_vertices_3d": hull_vertices_3d,
                })


        print(f"Layer {layer_num} segmentation completed \n \
        at a height of {layer_height} metres \n \
        {len(new_clusters)} trees found on this layer")

        overlap_threshold = 0.4
        all_clusters = merge_clusters(new_clusters, all_clusters, overlap_threshold)

        if view_clsuters:
            visualise_segments = []
            visualise_convex_hulls = []

            for cluster in all_clusters:
                # Visualizing clusters
                cluster_pnt_cld = o3d.geometry.PointCloud()
                cluster_pnt_cld.points = o3d.utility.Vector3dVector(cluster["points"])
                cluster_pnt_cld.colors = o3d.utility.Vector3dVector([cluster["color"]] * len(cluster["points"]))
                visualise_segments.append(cluster_pnt_cld)
            
            for hull_vertices in all_hulls:
                # Visualizing convex hull
                # hull_vertices = cluster["hull_vertices_3d"]
                hull_edges = [(i, (i + 1) % len(hull_vertices)) for i in range(len(hull_vertices))]

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(hull_vertices)
                line_set.lines = o3d.utility.Vector2iVector(hull_edges)
                line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(hull_edges))
                visualise_convex_hulls.append(line_set)

            visualise_segments.extend(visualise_convex_hulls)


    o3d.visualization.draw_geometries(visualise_segments, window_name="segmentation")





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
    points = classify_ground(points, visualise = False)


    # #---------------SEGMENTATION-----------------------
    
    layer_stacking(points, view_clsuters  = True)

    #---------------OTHER---------------------------
    # view_raw_cloud(points)




    
main()