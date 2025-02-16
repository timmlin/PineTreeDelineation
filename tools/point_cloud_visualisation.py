import laspy
import numpy as np
import open3d as o3d
import os
import threading
import multiprocessing


def create_grid_faces(grid_bounds, color=(1, 0, 0)):
    """
    Create grid lines and as Open3D LineSet for visualization.

    :param grid_bounds: List of tuples defining grid boundaries [(xmin, xmax, ymin, ymax, zmin, zmax), ...].
    :param color: RGB tuple for face color (default is red).
    :return: A tuple of Open3D LineSet objects.
    """
    points = []
    lines = []
    faces = []
    for coords in grid_bounds:
        for point in coords:
            xmin = point[0]
            xmax = point[1]
            ymin = point[2]
            ymax = point[3]
            zmin = point[4]
            zmax = point[5] 
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
            offset = len(points) - 8  # Offset for indexing the current box
            lines.extend([(a + offset, b + offset) for a, b in face_lines])

            # Define the vertical faces using triangles
            vertical_faces = [
                (0, 1, 5), (0, 5, 4),  # Side 1
                (1, 2, 6), (1, 6, 5),  # Side 2
                (2, 3, 7), (2, 7, 6),  # Side 3
                (3, 0, 4), (3, 4, 7),  # Side 4
            ]
            faces.extend([(a + offset, b + offset, c + offset) for a, b, c in vertical_faces])

        # Create LineSet for grid lines
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
        
    return line_set


def visulaise_las(input_file):
    """
    Visualize the point cloud with overlaid grid lines and vertical faces.

    :param input_file: Path to the input point cloud file (.las or .laz).
    """
    las = laspy.read(input_file)

    points = np.vstack((las.x, las.y, las.z)).T

    points = points[points[:, 0] >= 0] # x value threashold  
    points = points[points[:, 1] >= 0] # y value threashold  
    points = points[points[:, 2] >= 0] # z value threashold  

    min_x, min_y, min_z = points.min(axis=0)
    max_x, max_y, max_z = points.max(axis=0)

    boundaries = [  
                    [(min_x, max_x, 385, max_y, min_z, max_z)],
                    [(min_x, max_x, 385, 385, min_z, max_z)],
                    
                ]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    grid_lines = create_grid_faces(boundaries)
    
    np.set_printoptions(suppress=True)

    print(len(points))  
    print("Point cloud bounds:", np.min(points, axis=0), np.max(points, axis=0))
    print("Grid lines bounds:", np.min(grid_lines.points, axis=0), np.max(grid_lines.points, axis=0))


    o3d.visualization.draw_geometries(
        [point_cloud],
        window_name=f"{input_file} Point Cloud"
    )



def run_visualization(las_file_1, las_file_2):
    """
    Run two separate visualizations for las comaprison using multiprocessing.
    """
    process_1 = multiprocessing.Process(target=visulaise_las, args=(las_file_1,))
    process_2 = multiprocessing.Process(target=visulaise_las, args=(las_file_2,))

    process_1.start()
    process_2.start()

    process_1.join()
    process_2.join()






if __name__ == "__main__":
    
    las_file_path = r"data/UAV_sample_data/plot_31_pointcloud_normalised.las"

    visulaise_las(las_file_path)