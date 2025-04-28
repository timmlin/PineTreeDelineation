import laspy
import numpy as np
import open3d as o3d
import random
import time
import os
import subprocess
import sys
import re

from tools.utils import *



def view_classifications(las_file_path):
    # Load the LAS file
    las = laspy.read(las_file_path)
    
    # Extract point coordinates
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Extract the classification of each point
    classifications = las.classification

    # Assign colors based on classification
    colors = np.zeros((points.shape[0], 3))  # Initialize an array for the colors (RGB)

    # Classify the points into ground and non-ground
    # Assuming class 2 is ground and other values represent non-ground
    ground_mask = classifications == 1
    non_ground_mask = ~ground_mask

    colors[ground_mask] = [1, 0, 0]  # Ground points are red
    colors[non_ground_mask] = [0, 1, 0]  # Non-ground points are green

    # Create an Open3D point cloud and assign points and colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def view_raw_cloud(filename):

    # Load the LAS file
    las = laspy.read(filename)

    points = noramlise_las(las)
    
    # Extract point coordinates
    points = np.vstack((las.x, las.y, las.z)).transpose()


    # Function to visualize the point cloud, replace this with actual visualization code as needed
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pnt_cld])



view_raw_cloud('data/4D_7_2_03b_trans2.las')

