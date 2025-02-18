import laspy
import numpy as np
import open3d as o3d
import threading
import multiprocessing
import random

from tools.utils import *




#---------------------SEGMENTATION----------------------------


def layer_stacking(pnt_cld, layer_height = 1):
    """
    performs the layer stacking segmentation algorithm.

    Args:
        pnt_cld (open3d.cuda.pybind.geometry.PointCloud): point cloud to be segmented.

    Returns:
        return_type: Description of the return value.

    """
    tree_points = np.asarray(pnt_cld[0].points)
    ground_points = np.asarray(pnt_cld[1].points)

    #list to save segmented clouds
    clouds = []

    min_height = np.min(tree_points[:, 2])
    max_height = np.max(tree_points[:, 2])  

    print(min_height, max_height)

    num_layers = int(np.ceil((max_height - min_height) / layer_height))

    #list to store each layer
    layers = []

    for i in range(num_layers):
        
        cur_layer_min_height = i * layer_height
        cur_layer_max_height = cur_layer_min_height + layer_height

        layer_mask = (tree_points[:,2] >= cur_layer_min_height) & (tree_points[:,2] < cur_layer_max_height)

        layer_points = tree_points[layer_mask]

        layer_colour = [random.random(), random.random(),random.random()]
        layer_cloud = o3d.geometry.PointCloud()
        layer_cloud.points = o3d.utility.Vector3dVector(layer_points)
        layer_cloud.colors = o3d.utility.Vector3dVector([layer_colour] * len(layer_points))

        layers.append(layer_cloud)

    layers.append(pnt_cld[0])

    o3d.visualization.draw_geometries(layers, window_name = "layers")


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

    layer_stacking(pnt_cld)

main()