import laspy
import numpy as np
import open3d as o3d

from utils import *
import matplotlib.pyplot as plt



def view_segmentation(las_file_path):
    """
    Visualizes a segmented LAS point cloud with treeID coloring using Open3D.

    Args:
        las_file_path (str): Path to the segmented LAS file. Must contain a 'treeID' field.

    Returns:
        None
    """
    las = laspy.read(las_file_path)
    las = offset_to_origin(las)
    points = np.vstack((las.x, las.y, las.z)).transpose()


    # Check for treeID field
    if "treeID" not in las.point_format.extra_dimension_names:
        raise ValueError("No 'treeID' field found in LAS file.")

    tree_ids = las["treeID"]
    unique_ids = np.unique(tree_ids)

    # Assign a unique color to each treeID
    cmap = plt.get_cmap("tab20", len(unique_ids))
    color_dict = {tree_id: cmap(i % cmap.N)[:3] for i, tree_id in enumerate(unique_ids)}

    colors = np.array([color_dict[tid] for tid in tree_ids])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])






def view_raw_cloud(filename, z_threshold=0):
    """
    Visualizes a raw LAS point cloud using Open3D, optionally filtering to the top z_threshold meters.

    Args:
        filename (str): Path to the raw LAS file.
        z_threshold (float, optional): If set, only points within the top z_threshold meters are shown. Default is 0 (show all).

    Returns:
        None
    """
    # Load the LAS file
    las = laspy.read(filename)
    las = offset_to_origin(las)  # Assuming this is a user-defined function

    # Extract point coordinates
    points = np.vstack((las.x, las.y, las.z)).transpose()

    if z_threshold:
        max_z = np.max(points[:, 2])
        # Keep only the top z_threshold meters
        mask = points[:, 2] >= (max_z - z_threshold)
        points = points[mask]

    # Create the point cloud
    pnt_cld = o3d.geometry.PointCloud()
    pnt_cld.points = o3d.utility.Vector3dVector(points)

    # Set up the visualizer with black background
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pnt_cld)

    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # Black

    vis.run()
    vis.destroy_window()





