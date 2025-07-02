
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

