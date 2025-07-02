import laspy
import numpy as np
import open3d as o3d
import random
import time
import os
import subprocess
import sys
import re

from utils import *
import matplotlib.pyplot as plt



def view_segmentation(las_file_path):
    las = laspy.read(las_file_path)
    las = noramlise_las(las)
    points = np.vstack((las.x, las.y, las.z)).transpose()


    # Check for treeID field
    if "treeID" not in las.point_format.extra_dimension_names:
        raise ValueError("No 'treeID' field found in LAS file.")

    tree_ids = las["treeID"]
    unique_ids = np.unique(tree_ids)

    # Assign a unique color to each treeID
    cmap = plt.get_cmap("tab20", len(unique_ids))
    color_dict = {tree_id: cmap(i % cmap.N)[:3] for i, tree_id in enumerate(unique_ids)}

    colors = np.array([color_dict[tid] for tid in tree_ids])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])





def view_raw_cloud(filename, z_threshold=0):
    # Load the LAS file
    las = laspy.read(filename)
    las = noramlise_las(las)
    # Extract point coordinates
    points = np.vstack((las.x, las.y, las.z)).transpose()

    if z_threshold:
        max_z = np.max(points[:, 2])

        # Keep only the top z_threshold meters
        mask = points[:, 2] >= (max_z - z_threshold)
        points = points[mask]


    # Create and visualize the point cloud
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pnt_cld])






