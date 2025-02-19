import laspy
import numpy as np
import open3d as o3d
import threading
import multiprocessing
import random

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

    min_height = np.min(tree_points[:, 2])
    max_height = np.max(tree_points[:, 2])  

    print(min_height, max_height)

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


    


def main():

    las_file = "data/UAV_sample_data/plot_31_pointcloud.las"
    las = laspy.read(las_file)

    las = noramlise_las(las)

    points = np.vstack((las.x, las.y, las.z)).T

    # Convert to Open3D Point Cloud
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)

    # pnt_cld, _ = pnt_cld.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # pnt_cld = ransac_classify_ground(pnt_cld, visualise= True)
    pnt_cld = classify_ground(pnt_cld, visualise= False)

    layer_stacking(pnt_cld, visualise = True)

main()