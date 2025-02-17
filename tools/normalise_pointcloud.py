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

    # Plane segmentation to detect ground
    plane_model, inliers = pnt_cld.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)

    # Get ground plane equation (ax + by + cz + d = 0)
    a, b, c, d = plane_model

    # Extract ground points and non-ground points
    ground_points = np.asarray(pnt_cld.points)[inliers]
    non_ground_points = np.asarray(pnt_cld.points)[~np.isin(np.arange(len(pnt_cld.points)), inliers)]

    # Normalize ground by setting Z to 0
    ground_points[:, 2] -= (-d / c)

    # Merge ground and non-ground points back
    normalized_points = np.vstack((ground_points, non_ground_points))

    # Update the Open3D point cloud with normalized points
    normalized_pnt_cld = o3d.geometry.PointCloud()
    normalized_pnt_cld.points = o3d.utility.Vector3dVector(normalized_points)

    # Set the color for ground points
    ground_color = [1, 0, 0]  # Red color for ground
    ground_pnt_cld = o3d.geometry.PointCloud()
    ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
    ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_color] * len(ground_points))

    # Set the color for non-ground points
    non_ground_color = [0, 0, 1]  # Blue color for non-ground
    non_ground_pnt_cld = o3d.geometry.PointCloud()
    non_ground_pnt_cld.points = o3d.utility.Vector3dVector(non_ground_points)
    # non_ground_pnt_cld.colors = o3d.utility.Vector3dVector([non_ground_color] * len(non_ground_points))


    # Visualize the ground and non-ground point clouds with different colors
    print("Visualizing ground (red) and non-ground (blue) point clouds...")
    o3d.visualization.draw_geometries([ground_pnt_cld, non_ground_pnt_cld], window_name="Ground and Non-Ground Points")

def main():

    las_file = "data/UAV_sample_data/plot_31_pointcloud.las"
    las = laspy.read(las_file)

    las = noramlise_las(las)

    points = np.vstack((las.x, las.y, las.z)).T

    # Convert to Open3D Point Cloud
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)

    ransac_classify_ground(pnt_cld, visualise= True)

    

main()