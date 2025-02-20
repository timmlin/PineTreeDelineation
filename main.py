import laspy
import numpy as np
import open3d as o3d
import threading
import multiprocessing
import random
from sklearn.cluster import MeanShift


from tools.utils import *




#---------------------SEGMENTATION----------------------------


def layer_stacking(points, layer_height = 1, visualise = False):
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

    #list to store each layer
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

#-----------------------------------------------------------------------------------
    # visuals = []
    # to_cluster = np.vstack((layers[0], layers[1], layers[2]))

    # cluster = MeanShift(bandwidth=2).fit(to_cluster)
    
    # clustered_layer = []
    # clustered_layer.append(cluster)

    # centroids = clustered_layer[0].cluster_centers_


    # centroids_colour = [1, 0, 0] #red
    # centroids_pnt_cld = o3d.geometry.PointCloud()
    # centroids_pnt_cld.points = o3d.utility.Vector3dVector(centroids)
    # centroids_pnt_cld.colors = o3d.utility.Vector3dVector([centroids_colour] * len(centroids))

    # labels = cluster.labels_

    # unique_labels = np.unique(labels)
    # clustered_point_clouds = []

    # for label in unique_labels:
    #     # Select points belonging to the current cluster
    #     cluster_points = to_cluster[labels == label]
        
    #     # Create a point cloud for the current cluster
    #     cluster_pnt_cld = o3d.geometry.PointCloud()
    #     cluster_pnt_cld.points = o3d.utility.Vector3dVector(cluster_points)
        
    #     # Assign a unique color for each cluster (e.g., random color or a fixed one)
    #     cluster_color = np.random.rand(3)  # Random color
    #     cluster_pnt_cld.colors = o3d.utility.Vector3dVector([cluster_color] * len(cluster_points))
        
    #     clustered_point_clouds.append(cluster_pnt_cld)

    # # Visualize centroids and clusters
    # visuals.append(centroids_pnt_cld)
    # visuals.extend(clustered_point_clouds)
    # o3d.visualization.draw_geometries(visuals)










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


    #---------------SEGMENTATION-----------------------
    layer_stacking(points, visualise = True)

    #---------------OTHER---------------------------
    # view_raw_cloud(points)








    
main()