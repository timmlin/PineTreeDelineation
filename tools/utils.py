import laspy
import numpy as np
import open3d as o3d
import os

from sklearn.linear_model import RANSACRegressor
from scipy.spatial import cKDTree

#-------------------------PRE--PROCESSING----------------------

def offset_to_origin(las):
    """
    Shifts a point cloud so that the minimum x, y, z values are at 0 (origin).

    Args:
        las (laspy.lasdata.LasData): An opened LAS file data.

    Returns:
        laspy.lasdata.LasData: Shifted LAS data.
    """
    x_min, y_min, z_min = las.x.min(), las.y.min(), las.z.min()

    shifted_x = las.x - x_min
    shifted_y = las.y - y_min
    shifted_z = las.z - z_min

    # Update the LAS header to adjust the offset
    las.header.offsets = [0.0, 0.0, 0.0]  # Set the new offsets for X, Y, Z
    las.header.scales = [0.01, 0.01, 0.01]  # Keep scales small to retain precision

    las.x = shifted_x
    las.y = shifted_y
    las.z = shifted_z

    return las



def las_summary(filename):
    """
    Prints a summary of a LAS file, including format, point count, extent, CRS, area, and density.

    Args:
        filename (str): Path to the LAS file.

    Returns:
        None
    """
    las = laspy.read(filename)
    print(f"Filename: {filename}")
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

#------------------GROUND-CLASSIFICATION---------------------


def classify_ground_threshold(points, z_threshold, visualise=False):
    """
    Classifies the ground points based on the z-value (height) of points.

    Args:
        points (numpy.ndarray): Array of shape (N, 3) representing the point cloud.
        z_threshold (float): Height threshold for ground classification.
        visualise (bool, optional): If True, visualizes the result. Default is False.

    Returns:
        list: A list containing the non-ground and ground points arrays.
    """

    clouds = []


    # points = np.asarray(pnt_cld.points)
    
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
        o3d.visualization.draw_geometries([ground_pnt_cld, non_ground_pnt_cld])

    return clouds

def ransac_classify_ground(points, threshold=0.40, visualise=False):
    """
    Classifies the ground points of an Open3D Point Cloud using the RANSAC algorithm.

    Args:
        points (numpy.ndarray): Array of shape (N, 3) representing the point cloud.
        threshold (float): Maximum distance from the plane to classify a point as ground.
        visualise (bool, optional): If True, visualizes the result. Default is False.

    Returns:
        list: A list containing two numpy arrays: ground points and non-ground points.
    """
    
    #checks the bottom 30% of points for ground
    points_sorted = points[points[:, 2].argsort()]
    possible_ground_points = points_sorted[: int(len(points) * 0.3)]

    x = possible_ground_points[:, :2]  
    z = possible_ground_points[:, 2]   

    ransac = RANSACRegressor()
    ransac.fit(x, z)
    prediction = ransac.predict(x)

    distances = np.abs(z - prediction)

    ground_mask = distances < threshold

    # Predict ground for the whole point cloud using the RANSAC model
    prediction_full = ransac.predict(points[:, :2] )

    # Compute the distance between all points and the fitted plane
    distances_full = np.abs(points[:, 2] - prediction_full)

    # Create a mask for ground points in the whole point cloud
    ground_mask_full = distances_full < threshold

    ground_points = points[ground_mask_full]
    non_ground_points = points[~ground_mask_full]

    # Store ground and non-ground points in a list
    clouds = [ground_points, non_ground_points]

    if visualise:
        ground_color = [1, 0, 0] # Red 

        non_ground_color = [0, 0, 1] #Blue 

        # Create Open3D point clouds for visualization
        ground_pnt_cld = o3d.geometry.PointCloud()
        ground_pnt_cld.points = o3d.utility.Vector3dVector(ground_points)
        ground_pnt_cld.colors = o3d.utility.Vector3dVector([ground_color] * len(ground_points))

        non_ground_pnt_cld = o3d.geometry.PointCloud()
        non_ground_pnt_cld.points = o3d.utility.Vector3dVector(non_ground_points)
        non_ground_pnt_cld.colors = o3d.utility.Vector3dVector([non_ground_color] * len(non_ground_points))

        # Visualize the classified point clouds
        o3d.visualization.draw_geometries([ground_pnt_cld, non_ground_pnt_cld])

    return clouds




def save_segmented_las(clusters, ground_points, filename):
    """
    Saves segmented clusters and ground points to a LAS file, assigning treeID per cluster.

    Args:
        clusters (list): List of cluster dicts, each with a 'points' key (N, 3 array).
        ground_points (numpy.ndarray): Array of ground points.
        filename (str): Output LAS file name.

    Returns:
        None
    """
    # Flatten clustered points and assign treeID per cluster
    cluster_points = []
    tree_ids = []

    for tree_id, cluster in enumerate(clusters, start=1):
        cluster_points.append(cluster["points"])
        tree_ids.extend([tree_id] * len(cluster["points"]))

    cluster_points = np.vstack(cluster_points)
    tree_ids = np.array(tree_ids, dtype=np.uint32)

    # Combine with ground points; assign treeID = 0 for ground
    all_points = np.vstack((cluster_points, ground_points))
    all_tree_ids = np.concatenate((tree_ids, np.zeros(len(ground_points), dtype=np.uint32)))

    # Define LAS header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(all_points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    # Create LasData and assign standard fields
    las = laspy.LasData(header)
    las.x = all_points[:, 0]
    las.y = all_points[:, 1]
    las.z = all_points[:, 2]

    # Add 'treeID' as an extra dimension
    if "treeID" not in las.point_format.extra_dimension_names:
        treeid_dim = laspy.ExtraBytesParams(name="treeID", type=np.uint32)
        las.add_extra_dim(treeid_dim)

    las["treeID"] = all_tree_ids

    las.write(filename)


