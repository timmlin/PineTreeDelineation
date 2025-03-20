import laspy
import numpy as np
import open3d as o3d
import random
import time
import os
import subprocess
import sys
import re

from sklearn.cluster import DBSCAN

from tools.utils import *
from tools.layer_stacking import layer_stacking



def view(filename):

    las = laspy.read(filename)
    las = noramlise_las(las)

    points = np.vstack((las.x, las.y, las.z)).T
 
    # Convert to Open3D Point Cloud to remove noise
    pnt_cld = o3d.geometry.PointCloud() 
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    pnt_cld, _ = pnt_cld.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    #Convert back to np array
    points = np.asarray(pnt_cld.points)
    view_raw_cloud(points)


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




def adjust_ground_height(filename):
    # Read the .las file using laspy
    las = laspy.read(filename)
    
    # Extract the ground points based on the classification (Ground is classified as 1)
    ground_mask = las.classification == 1  # Ground points are classified as 1
    
    # Convert the point cloud to a numpy array for manipulation
    points = np.vstack((las.x, las.y, las.z)).T
    
    # Get the original ground height (Z-value) before any adjustment
    original_ground_height = np.mean(points[ground_mask, 2])
    
    # Set the Z values of ground points to 0
    points[ground_mask, 2] = 0
    
    # Adjust the entire point cloud by subtracting the original ground height
    # Only adjust the points that are not ground
    points[~ground_mask, 2] -= original_ground_height  # Subtract the original ground height from non-ground points


    # Return or visualize the adjusted point cloud (you can replace view_raw_cloud with your display function)
    view_raw_cloud(points)

    print(min(points[2]))
def view_raw_cloud(points):
    # Function to visualize the point cloud, replace this with actual visualization code as needed
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pnt_cld])


adjust_ground_height('data/SCION/csf_ground/tile_7_9.las')