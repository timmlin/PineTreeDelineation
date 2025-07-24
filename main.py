import laspy
import numpy as np
import open3d as o3d

from tools.utils import *
from src.layered_clusters import *


def main():
    """
    Example workflow for layered clustering segmentation on a LAS file.

    Args:
        None

    Returns:
        None
    """
    # -------------INPUT-LAS-FILE-------------
    file_path = 'path-to-input-las-file.las'

    las = laspy.read(file_path)

    #-------------PRE-PROCESSING
    las = offset_to_origin(las)

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

    points, ground_points = layered_clusters(points, view_layers=False, view_clusters=True)



    save_segmented_las(points, ground_points)





if __name__ == "__main__":
    main()