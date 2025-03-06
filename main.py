import laspy
import numpy as np
import open3d as o3d
import random

from sklearn import datasets, linear_model
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

from tools.utils import *

from tools.layer_stacking import layer_stacking








def main():

    plot_31_las_file = "data/UAV_sample_data/plot_31_pointcloud.las"
    plot_87_las_file = 'data/SCION/plot_87_annotated.las'
    las = laspy.read(plot_87_las_file)
    #-------------PRE-PROCESSING
    
    las = noramlise_las(las)

    points = np.vstack((las.x, las.y, las.z)).T

    # Convert to Open3D Point Cloud to remove noise
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    pnt_cld, _ = pnt_cld.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    #Convert back to np array
    points = np.asarray(pnt_cld.points)

    #---------------OTHER---------------------------
    # view_raw_cloud(points)
    
    #---------------GROUND-CLASSIFICATION--------------
    # points = ransac_classify_ground(pnt_cld, visualise= True)
    points = classify_ground_threshold(points, visualise = False)
    # points = classify_ground_csf(points)


    #---------------SEGMENTATION-----------------------    
    layer_stacking(points, view_clsuters  = True)








    
main()