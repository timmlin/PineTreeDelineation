import laspy
import numpy as np
import open3d as o3d
import random
import time
import os
import sys
import re


from tools.utils import *

def las_summary(filename):

    las = laspy.read(filename)
    
    print(f"Format Version: {las.header.version}")
    print(f"Point Format ID: {las.header.point_format.id}")
    print(f"Number of Points: {len(las.points)}")

    xmin, ymin, zmin = las.header.min
    xmax, ymax, zmax = las.header.max
    print(f"Extent: xmin={xmin:.2f}, xmax={xmax:.2f}, ymin={ymin:.2f}, ymax={ymax:.2f}")

    # Coordinate Reference System (CRS) (if available)
    if "WKT" in las.header.vlrs[0].description:
        print(f"Coordinate Reference System: {las.header.vlrs[0].record_data}")

    area = (xmax - xmin) * (ymax - ymin)
    print(f"Area: {area:.2f} m²")

    num_points = len(las.points)
    density = num_points / area
    print(f"Point Density: {density:.2f} points/m²")




def get_ground_execution_times(filename):

    #read file
    start_read = time.time()
    las = laspy.read(filename)
    end_read = (time.time() - start_read)
    print(f"file {filename} read in {end_read:.2f} seconds")
    
    las_summary(filename)


    #remove outliers
    start_outlier = time.time()
    las = noramlise_las(las)

    points = np.vstack((las.x, las.y, las.z)).T

    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    pnt_cld, _ = pnt_cld.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    points = np.asarray(pnt_cld.points)
    end_outlier = (time.time() - start_outlier)
    print(f"outliers removed in {end_outlier:.2f} seconds")
    


    #RANSAC
    start_ransac = time.time()
    ransac_classify_ground(points, 1000, visualise = False)
    end_ransac = (time.time() - start_ransac)
    print(f"ground found using RANSAC in {end_ransac:.2f} seconds")

    start_threshold = time.time()
    points = classify_ground_threshold(points, 2, visualise = False)
    end_threshold = (time.time() - start_threshold)
    print(f"ground found using thresholding in {end_threshold:.2f} seconds")



directory = 'data/Rolleston Forest plots'

with open("groud_eval.txt", 'w') as output_file:
    sys.stdout = output_file

    for filename in os.listdir(directory):
        if re.match(r'^plot(?:_\d+)+\.las$', filename):
            file_path = os.path.join(directory, filename)

            if os.path.isfile(file_path):
                get_ground_execution_times(file_path)
                print()

