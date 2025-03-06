import laspy
import numpy as np
import open3d as o3d

from tools.utils import *

def main():

    las_file = "data/SCION/plot_87_annotated.las"
    las = laspy.read(las_file)

    #-------------PRE-PROCESSING
    
    las = noramlise_las(las)

    points = np.vstack((las.x, las.y, las.z)).T

    # Convert to Open3D Point Cloud to remove noise
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    pnt_cld, _ = pnt_cld.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    #Convert back to np array
    points = np.asarray(pnt_cld.points)

    view_raw_cloud(points)

    
main()