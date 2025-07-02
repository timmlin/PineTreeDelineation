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
from layered_clusters import layered_clusters, save_classified_las

from graph_segmentation import graph


def main():


    tile_num = 

    file_path = f'data/results/layered_clusters/rolleston_segmented_las/tile_{tile_num}_las/tile_{tile_num}_segmented.las'



    las = laspy.read(file_path)

    #-------------PRE-PROCESSING
    las = noramlise_las(las)

    points = np.vstack((las.x, las.y, las.z)).T

    # Convert to Open3D Point Cloud to remove noise
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    pnt_cld = pnt_cld.voxel_down_sample(voxel_size=0.2)
    pnt_cld, _ = pnt_cld.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    points = np.asarray(pnt_cld.points)

    #---------------GROUND-CLASSIFICATION--------------
    points = classify_ground_threshold(points, 1, visualise=False)

    #--------------SEGMENTATION-----------------------
    start_time = time.time()

    points, ground_points = layered_clusters(points, view_layers=False, view_clusters=True)

    end_time = time.time()
    total_time = end_time - start_time




main()
