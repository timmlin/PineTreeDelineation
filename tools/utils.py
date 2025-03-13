import laspy
import numpy as np
import open3d as o3d
import os

from sklearn.linear_model import RANSACRegressor
from sklearn.svm import SVR
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

def svr_classify_ground(points, threshold = 0.2, visualise = False):

    #checks the bottom 30% of points for ground
    points_sorted = points[points[:, 2].argsort()]
    possible_ground_points = points_sorted[: int(len(points) * 0.3)]

    x = possible_ground_points[:, :2]  
    z = possible_ground_points[:, 2]   

    svr = SVR(kernel='rbf')   
    svr.fit(x,z)
    prediction = svr.predict(x)

    distances = np.abs(z - prediction)

    ground_mask = distances < threshold

    # Predict ground for the whole point cloud using the SVR model
    prediction_full = svr.predict(points[:, :2] )

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



#---------------------MISC------------------------
def save_as_las_file(points, file_name):
    """saves the np array of points as an las file"""

    header = laspy.LasHeader(point_format=3)

    las_data = laspy.LasData(header)

    las_data.x = points[:,0]
    las_data.y = points[:,1]
    las_data.z = points[:,2]

    las_data.write(file_name)

























    
def view_raw_cloud(points):
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)
    # pnt_cld.colors = o3d.utility.Vector3dVector([centroids_colour] * len(centroids))
    o3d.visualization.draw_geometries([pnt_cld], window_name="Point Cloud")



def divide_laz_into_grid(input_laz, output_folder, grid_size):
    """Divides a .laz file into a grid and saves each grid cell as a separate file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    las = laspy.read(input_laz)
    points = np.vstack((las.x, las.y, las.z)).T
    
    #  grid boundaries
    min_x, max_x = np.min(las.x), np.max(las.x)
    min_y, max_y = np.min(las.y), np.max(las.y)
    
    x_intervals = np.arange(min_x, max_x, grid_size)
    y_intervals = np.arange(min_y, max_y, grid_size)
    
    for i, x_start in enumerate(x_intervals):
        for j, y_start in enumerate(y_intervals):
            x_end = x_start + grid_size
            y_end = y_start + grid_size
            
            # Select points within the grid cell
            mask = (las.x >= x_start) & (las.x < x_end) & (las.y >= y_start) & (las.y < y_end)
            
            if np.any(mask):
                sub_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
                sub_las.points = las.points[mask]
                
                output_filename = os.path.join(output_folder, f"tile_{i}_{j}.laz")
                sub_las.write(output_filename)
                print(f"Saved: {output_filename}")

