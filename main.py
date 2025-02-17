import laspy
import numpy as np
import open3d as o3d
import threading
import multiprocessing

from tools.normalise_pointcloud import *

def main():

    las_file = "data/UAV_sample_data/plot_31_pointcloud.las"
    las = laspy.read(las_file)

    las = noramlise_las(las)

    points = np.vstack((las.x, las.y, las.z)).T

    # Convert to Open3D Point Cloud
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)



    pnt_cld = ransac_classify_ground(pnt_cld, visualise= True)

    

main()