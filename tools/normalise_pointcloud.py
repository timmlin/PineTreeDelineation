import laspy
import numpy as np
import open3d as o3d

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
    plane_model, inliers = pnt_cld.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)


    # Extract ground points and non-ground points
    ground_points = np.asarray(pnt_cld.points)[inliers]
    non_ground_points = np.asarray(pnt_cld.points)[~np.isin(np.arange(len(pnt_cld.points)), inliers)]
    print(type(ground_points))

    # Merge ground and non-ground points back
    normalized_points = np.vstack((ground_points, non_ground_points))

    normalized_pnt_cld = o3d.geometry.PointCloud()
    normalized_pnt_cld.points = o3d.utility.Vector3dVector(normalized_points)

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


    # Visualize the ground and non-ground point clouds with different colors
    print("Visualising ground classification ")
    o3d.visualization.draw_geometries(clouds, window_name="Ground Classification")



def classify_ground(pnt_cld, visualise = False):
    """
    classifies the ground points of an Open3D Point Cloud based on a points z value

    Args:
        pnt_cld (open3d.cuda.pybind.geometry.PointCloud): an Open3D point cloud.
        visulaise (bool): flag that determines if the point cloud will be visulaised afer classification

    Returns:
        las open3d.cuda.pybind.geometry.PointCloud: an Open3D point cloud with classified ground points.

    """
    clouds = []


    points = np.asarray(pnt_cld.points)

    z_threshold = 5
    
    ground_mask = points[:, 2] <= z_threshold
    non_ground_points = points[:, 2] > z_threshold


    # Assign ground points a different color (e.g., red for visualization)
    colors = np.zeros_like(points)  # Default color black
    colors[ground_mask] = [1, 0, 0]  # Red for ground points
    
    # Update point cloud colors
    pnt_cld.colors = o3d.utility.Vector3dVector(colors)
    
    if visualise:
        o3d.visualization.draw_geometries([pnt_cld], window_name="Classified Ground Points")
    
    return pnt_cld
