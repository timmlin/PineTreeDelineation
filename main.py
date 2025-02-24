import laspy
import numpy as np
import open3d as o3d
import threading
import multiprocessing
import random
from sklearn.cluster import MeanShift
from scipy.spatial import ConvexHull


from tools.utils import *




#---------------------SEGMENTATION----------------------------


def layer_stacking(points, layer_height = 1, visualise = True):
    """
    performs the layer stacking segmentation algorithm.

    Args:
        pnt_cld (open3d.cuda.pybind.geometry.PointCloud): point cloud to be segmented.

    Returns:
        return_type: Description of the return value.

    """
    tree_points = points[0]
    ground_points = points[1]

    #list to save segmented clouds
    clouds = []

    print
    min_height = np.min(tree_points[:, 2])
    max_height = np.max(tree_points[:, 2])  

    # print(min_height, max_height)

    num_layers = int(np.ceil((max_height - min_height) / layer_height))


    #----------------------LAYERING-------------
    layers = []
    to_visulase = []
    for i in reversed(range(num_layers)):
        
        cur_layer_min_height = i * layer_height
        cur_layer_max_height = cur_layer_min_height + layer_height

        layer_mask = (tree_points[:,2] >= cur_layer_min_height) & (tree_points[:,2] < cur_layer_max_height)

        layer_points = tree_points[layer_mask]

        if visualise:
            layer_colour = [random.random(), random.random(),random.random()]
            layer_cloud = o3d.geometry.PointCloud()
            layer_cloud.points = o3d.utility.Vector3dVector(layer_points)
            layer_cloud.colors = o3d.utility.Vector3dVector([layer_colour] * len(layer_points))
            to_visulase.append(layer_cloud)

        layers.append(layer_points)

    layers.append(ground_points)



    if visualise: 
        # creates an open3d point cloud of ground points
        ground_color = [1, 0, 0] #red
        ground_pnt_cld = o3d.geometry.PointCloud()
        ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
        ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_color] * len(ground_points))
    
        to_visulase.append(ground_pnt_cld)        
        o3d.visualization.draw_geometries(to_visulase, window_name = "layers")

#---------------------------SEGMENTATION-------------------------------------
    visuals = []
    previous_centroids = None

    for i in range(1):
        to_cluster = np.vstack(layers[i])
        layer_height = np.mean(to_cluster[:, 2]) 
        print(layer_height)
            

        
        cluster = MeanShift(bandwidth=2).fit(to_cluster)
        
        labels = cluster.labels_
        unique_labels = np.unique(labels)
        centroids = cluster.cluster_centers_

        #plot the centroids 
        clustered_layer = []
        clustered_layer.append(cluster)

        centroids_colour = [1, 0, 0] #red
        centroids_pnt_cld = o3d.geometry.PointCloud()
        centroids_pnt_cld.points = o3d.utility.Vector3dVector(centroids)
        centroids_pnt_cld.colors = o3d.utility.Vector3dVector([centroids_colour] * len(centroids))


        clustered_point_clouds = []
        convex_hulls = []

        for label in unique_labels:
            cluster_points = to_cluster[labels == label]

            xy_points = cluster_points[:, :2]
            z_mean = np.mean(cluster_points[:, 2])
            
            if len(xy_points) >= 3:  # 2D convex hull requires at least 3 points
                hull = ConvexHull(xy_points)
                hull_vertices_xy = xy_points[hull.vertices]  

                # Create 3D points by mean z value
                hull_vertices = np.hstack([hull_vertices_xy, np.full((len(hull_vertices_xy), 1), z_mean)])

                # Create hull edges
                hull_edges = [(i, (i + 1) % len(hull.vertices)) for i in range(len(hull.vertices))]

                # Create Open3D lineset object for convex hull
                convex_hull = o3d.geometry.LineSet()
                convex_hull.points = o3d.utility.Vector3dVector(hull_vertices)
                convex_hull.lines = o3d.utility.Vector2iVector(hull_edges)
                convex_hull.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(hull_edges))  # green

                convex_hulls.append(convex_hull)

            
            # Create a point cloud for the current cluster
            cluster_pnt_cld = o3d.geometry.PointCloud()
            cluster_pnt_cld.points = o3d.utility.Vector3dVector(cluster_points)
            
            cluster_color = np.random.rand(3)
            cluster_pnt_cld.colors = o3d.utility.Vector3dVector([cluster_color] * len(cluster_points))
            
            clustered_point_clouds.append(cluster_pnt_cld)

        # Visualize centroids and clusters
        visuals.append(centroids_pnt_cld)
        visuals.extend(clustered_point_clouds)
        visuals.extend(convex_hulls)
        
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