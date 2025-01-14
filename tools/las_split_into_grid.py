import os
import numpy as np
import laspy

def split_las_into_grid(input_file, output_dir, grid_size=(7, 7)):
    # Read the LAS file
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T

    # Calculate point cloud bounds
    min_x, min_y, _ = points.min(axis=0)
    max_x, max_y, _ = points.max(axis=0)

    # Calculate cell dimensions
    x_step = (max_x - min_x) / grid_size[0]
    y_step = (max_y - min_y) / grid_size[1]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the grid
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Define cell boundaries
            min_x_cell = min_x + i * x_step
            max_x_cell = min_x + (i + 1) * x_step
            min_y_cell = min_y + j * y_step
            max_y_cell = min_y + (j + 1) * y_step

            # Filter points within the cell
            mask = (
                (las.x >= min_x_cell) & (las.x < max_x_cell) &
                (las.y >= min_y_cell) & (las.y < max_y_cell)
            )
            if np.sum(mask) == 0:
                continue  # Skip empty cells

            # Create a new LAS file with the filtered points
            new_las = laspy.LasData(las.header)
            new_las.points = las.points[mask]

            # Save the new LAS file
            output_file = os.path.join(output_dir, f"cell_{i}_{j}.las")
            new_las.write(output_file)

    print(f"Point cloud split into {grid_size[0]}x{grid_size[1]} grid and saved in {output_dir}.")




split_las_into_grid(r"data/Rolleston_lidar_20230707_normalised.las", "Rolleston_grid")