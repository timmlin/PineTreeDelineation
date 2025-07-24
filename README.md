# Machine Learning Based Pine Tree Segmentation
![Image](https://github.com/user-attachments/assets/bad41b6a-447f-4e83-982a-c6e510d692bd)
## Introduction

This project focuses on identifying individual pine trees in New Zealand forests using aerial laser scanning and machine learning. Four instance segmentation models were developed and evaluated over the course of this study.

The data is in the form of a 3D "point cloud" that represents the shape and structure of pine tree forests. Using this data, we apply and evaluate the performance of the four machine learning models to automatically detect and separate individual trees. 

By comparing different methods, we aim to find out which approach works best for different types of Zealand forest environments. The goal is to make forest monitoring faster, cheaper, and non-destructive, while still providing accurate results.


## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-results)
- [Requiremnets](#requirements)
- [Set Up](#setup)
- [Contact](#contact)
- [Example Usage](#example-usage)

## Overview

Four main segmentation models were implemented and tested:

1. **Layered Clustering** – a custom method combining DBSCAN and K-Means over horizontal layers of a normalised point cloud.
2. **Li2012 Algorithm** – a region-growing approach applied top-down on 3D point clouds.
3. **Dalponte2016** – uses a CHM with local maxima filtering followed by 3D region growing.
4. **Dalponte + DBSCAN** – a hybrid model combining DBSCAN-based tree top detection with the Dalponte2016 region-growing step.

Performance was evaluated on:
- **Rolleston Trial Forest**: Flat terrain, uniform pine structure.
- **Mohaka Forest**: Sloped terrain with dense and varied canopy.




## Key Findings

| Model              | Relative Error (Rolleston) | F1 Score (Mohaka) | Avg. Execution Time |
|--------------------|---------------|------------------|----------------------|
| Layered Clustering | 14.53%        | 0.27             | **7.69s**     |
| Li2012             | 17.79%        | 0.81             | 470.63s   |
| Dalponte2016       | 18.27%        | **0.89**         | 32.90s    |
| Dalponte + DBSCAN  | **13.83%**    | 0.56             | 32.85s    |


Dalponte2016 performed best in complex terrain (Mohaka), while Dalponte+DBSCAN was most accurate in simpler conditions (Rolleston). Layered Clustering had the lowest computational cost. 



## Requirements

- Python 3.12.3
- R 4.5.0
- Python packages: `numpy`, `laspy`, `open3d`, `sklearn`  
- R packages: `lidR`, `future`, `RCSF`, `terra`, `tidyverse`, `dbscan`, `sf`



## Setup

1. Clone this repo:
```bash
   git clone https://github.com/timmlin/tree-delineation.git
   cd tree-delineation
```
2. Install reqirements
```py
    pip install numpy, laspy, open3d, sklearn
```
```R
    install.packages(c("lidR", "future", "RCSF", "terra", "tidyverse", "dbscan", "sf"))
```

## Project Structure

```
.
├── data/                # Input point cloud data (not included in repo)
├── outputs/             # Segmented output files
├── output_files/        # Additional output files
├── src/
│   ├── layered_clusters.py      # Layered Clustering implementation
│   ├── li_segmentation.R        # Li2012 segmentation script
│   ├── dalponte2016.R           # Dalponte2016 segmentation script
│   ├── dalponte_dbscan.R        # Dalponte+DBSCAN hybrid script
│   └── view.r                   # Visualization helpers
├── tools/
│   ├── utils.py                 # Python utility functions 
│   ├── ground_classification.py # Ground classification helpers
│   ├── view.py                  # Python visualization helpers
│   ├── ground_truth.r           # R ground truth helpers
│   └── evaluate_ground_truth.R  # R evaluation script
├── main.py              # Main Python entry point for Layered Clustering 
├── README.md
└── .gitignore
```


## Example Usage

### Python (Layered Clustering)
```bash
python main.py
```
- By default, this runs the layered clustering segmentation on a sample LAS file (`path-to-input-las-file`).
- Output segmented files will be saved in the `outputs/` directory.


### R (Li2012, Dalponte2016, Dalponte+DBSCAN)
Open R and run, for example:
```R
source("src/li_segmentation.R")
```
- Edit the script to point to your LAS file and desired algorithm.
- Output files will be saved in the `outputs/` directory.

## Results Visualization

You can visualize point clouds using the provided Python functions in `tools/view.py`:

- **View a raw point cloud:**
  ```python
  from tools.view import view_raw_cloud
  view_raw_cloud("raw_file.las")
  # Optionally, you can set a z_threshold (in meters) to view only the top part of the canopy:
  # view_raw_cloud("your_raw_file.las", z_threshold=10)
  ```

- **View a segmented point cloud:**
  ```python
  from tools.view import view_segmentation
  view_segmentation("segmented_file.las")
  # The LAS file must contain a 'treeID' field for segmentation coloring.
  ```

These functions use Open3D for interactive 3D visualization. Make sure your LAS files are normalized and, for segmentation, include the required 'treeID' attribute.

## Acknowledgements

- Data: Thanks to the University of Canterbury and collaborators for providing point cloud datasets.
- Libraries: [lidR](https://github.com/Jean-Romain/lidR), [open3d](http://www.open3d.org/), [laspy](https://laspy.readthedocs.io/), [scikit-learn](https://scikit-learn.org/), [tidyverse](https://www.tidyverse.org/), and others.

## Cite

If you use this code or methodology in your research, please cite:

> Lindbom, T. (2025). *An Evaluation of Machine Learning-Based Pine Tree Segmentation Techniques* University of Canterbury.

## Contact
Tim Lindbom - Linkedin: https://www.linkedin.com/in/tim-lindbom/ - email: lindbomtim@gmail.com
