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
from layered_clusters import layer_stacking
from watershed import watershed





def main(filename):

    start_time = time.time()
    

    las = laspy.read(filename)
    #-------------PRE-PROCESSING
    
    las = noramlise_las(las)

    points = np.vstack((las.x, las.y, las.z)).T

    # Convert to Open3D Point Cloud to remove noise
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    pnt_cld, _ = pnt_cld.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    #Convert back to np array
    points = np.asarray(pnt_cld.points)

    #---------------OTHER---------------------------
    # view_raw_cloud(points)
    
    #---------------GROUND-CLASSIFICATION--------------
    points = classify_ground_threshold(points, 1, visualise = False)


    tree_points, ground_points = points
    watershed(tree_points, ground_points)

main('data/Rolleston Forest plots/plot_45.las')