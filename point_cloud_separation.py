import laspy
import numpy as np
import open3d as o3d
import os


def create_grid_lines(grid_bounds, color=(1, 0, 0)):
    """
    Create grid lines as Open3D LineSet for visualization.

    :param grid_bounds: List of tuples defining grid boundaries [(xmin, xmax, ymin, ymax, zmin, zmax), ...].
    :param color: RGB tuple for grid line color (default is red).
    :return: Open3D LineSet object.
    """
    points = []
    lines = []

    for (xmin, xmax, ymin, ymax, zmin, zmax) in grid_bounds:
        corners = [
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],  # Bottom face
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],  # Top face
        ]
        points.extend(corners)

        face_lines = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        offset = len(points) - 8  
        lines.extend([(a + offset, b + offset) for a, b in face_lines])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


def visualize_with_grid(input_file):
    """
    Visualize the point cloud with overlaid grid lines.

    :param input_file: Path to the input point cloud file (.las or .laz).
    """

    las = laspy.read(input_file)
    
    # Extract points
    points = np.vstack((las.x, las.y, las.z)).T


    # Calculate boundaries
    min_x, min_y, min_z = points.min(axis=0)
    max_x, max_y, max_z = points.max(axis=0)

    boundries = [(min_x, max_x, min_y, max_y, min_z, max_z)]
    # test = [(1.54727448e+06, 5.17032271e+06, 5.43426585e+01, 1.54732540e+06, 5.17036603e+06, 7.78227585e+01)]
    
    # Create an Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)


    grid_lines = create_grid_lines(boundries)

    print(len(points))  # Check how many points are in the point cloud
    
    print("Point cloud bounds:", np.min(points, axis=0), np.max(points, axis=0))
    print("Grid lines bounds:", np.asarray(grid_lines.points).min(axis=0), np.asarray(grid_lines.points).max(axis=0))

    o3d.visualization.draw_geometries([grid_lines, point_cloud], window_name="Point Cloud with Grid")


    



if __name__ == "__main__":
    las_file_path = r"PineTreeDelineation/data/Rolleston_lidar_20230707.las"
    visualize_with_grid(las_file_path)


