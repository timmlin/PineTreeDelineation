import laspy
import numpy as np
import open3d as o3d


#-------------------------PRE--PROCESSING----------------------

def noramlise_las(las):
    """
    Normalises a point cloud (sets minimum x, y, z values to 0)

    Args:
        las (laspy.lasdata.LasData): an opened las file data.

    Returns:
        laspy.lasdata.LasData: normalised las data.

    """
    x_min, y_min, z_min = las.x.min(), las.y.min(), las.z.min()

    normalized_x = las.x - x_min
    normalized_y = las.y - y_min
    normalized_z = las.z - z_min

    # Update the LAS header to adjust the offset
    las.header.offsets = [0.0, 0.0, 0.0]  # Set the new offsets for X, Y, Z
    las.header.scales = [0.01, 0.01, 0.01]  # Keep scales small to retain precision

    las.x = normalized_x
    las.y = normalized_y
    las.z = normalized_z

    return las


def ransac_classify_ground(pnt_cld, visualise = False):
    """
    classifies the ground points of an Open3D Point Cloud using the RANSAC algorithm

    Args:
        pnt_cld (open3d.cuda.pybind.geometry.PointCloud): an Open3D point cloud.
        visulaise (bool): flag that determines if the point cloud will be visulaised afer classification

    Returns:
        las open3d.cuda.pybind.geometry.PointCloud: an Open3D point cloud with classified ground points.

    """


    clouds = []

    # Plane segmentation to detect ground
    plane_model, inliers = pnt_cld.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=1000)


    # Extract ground points and non-ground points
    ground_points = np.asarray(pnt_cld.points)[inliers]
    non_ground_points = np.asarray(pnt_cld.points)[~np.isin(np.arange(len(pnt_cld.points)), inliers)]
    print(type(ground_points))


    #creates an open3d point cloud of ground points
    ground_color = [1, 0, 0] #red
    ground_pnt_cld = o3d.geometry.PointCloud()
    ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
    ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_color] * len(ground_points))

    clouds.append(ground_pnt_cld)

    #creates an open3d point cloud of ground points
    non_ground_color = [0, 0, 1] 
    non_ground_pnt_cld = o3d.geometry.PointCloud()
    non_ground_pnt_cld.points = o3d.utility.Vector3dVector(non_ground_points)
    # non_ground_pnt_cld.colors = o3d.utility.Vector3dVector([non_ground_color] * len(non_ground_points))
    
    clouds.append(non_ground_pnt_cld)


    if visualise:
        o3d.visualization.draw_geometries([pnt_cld], window_name="Classified Ground Points")


def classify_ground(points, visualise=False):
    """
    Classifies the ground points of an Open3D Point Cloud based on the z-value (height) of points.

    Args:
        pnt_cld (open3d.cuda.pybind.geometry.PointCloud): An Open3D point cloud.
        visualise (bool, optional): If True, the classified point cloud will be visualized. Defaults to False.

    Returns:
        clouds (list): A list containing the ground and non ground points arrays.
    """

    clouds = []


    # points = np.asarray(pnt_cld.points)

    z_threshold = 1
    
    ground_points_mask = points[:, 2] <= z_threshold
    non_ground_points_mask = points[:, 2] > z_threshold

    ground_points = points[ground_points_mask]
    non_ground_points = points[non_ground_points_mask]

    clouds.append(non_ground_points)

    clouds.append(ground_points)
    
    
    if visualise:
    
        #creates an open3d point cloud of ground points
        non_ground_pnt_cld = o3d.geometry.PointCloud()
        non_ground_pnt_cld.points = o3d.utility.Vector3dVector(non_ground_points)
    
        # #creates an open3d point cloud of ground points
        ground_color = [1, 0, 0] #red
        ground_pnt_cld = o3d.geometry.PointCloud()
        ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
        ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_color] * len(ground_points))
    
        o3d.visualization.draw_geometries([ground_pnt_cld, non_ground_pnt_cld], window_name="Classified Ground Points")

    return clouds


def view_raw_cloud(points):
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    # pnt_cld.colors = o3d.utility.Vector3dVector([centroids_colour] * len(centroids))
    o3d.visualization.draw_geometries([pnt_cld], window_name="Classified Ground Points")
