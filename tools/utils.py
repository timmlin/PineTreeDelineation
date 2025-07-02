import laspy
import numpy as np
import open3d as o3d
import os

import geopandas as gpd
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import cKDTree

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



def las_summary(filename):

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

    Returns:
        clouds (list): A list containing the ground and non ground points arrays.
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
        points (numpy.ndarray): A numpy array of shape (N, 3) representing the point cloud,
                                with columns representing x, y, z coordinates.
        threshold (float): The maximum distance from the plane to classify a point as ground.
        visualize (bool): If True, visualizes the ground and non-ground points using Open3D.

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


def view_raw_cloud(points):
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    # pnt_cld.colors = o3d.utility.Vector3dVector([centroids_colour] * len(centroids))
    o3d.visualization.draw_geometries([pnt_cld], window_name="Point Cloud")



def save_segmented_las(clusters, ground_points, filename="segmented_output.las"):
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






def open_shp(filename):

    shp = gpd.read_file(filename)

    print(len(shp))


