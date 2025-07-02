# Machine Learning Based Pine Tree Segmentation
![Image](https://github.com/user-attachments/assets/bad41b6a-447f-4e83-982a-c6e510d692bd)
## Description
## Introduction

This project focuses on identifying individual pine trees in New Zealand forests using aerial laser scanning and machine learning. Four instance segmentation models were developed and evaluated over the course of this study.

The data is in the form of a 3D "point cloud" that represents the shape and structure of pine tree forests. Using this data, we apply and evaluate the performance of the four machine learning models to automatically detect and separate individual trees. 

By comparing different methods, we aim to find out which approach works best for different types of Zealand forest environments. The goal is to make forest monitoring faster, cheaper, and non-destructive, while still providing accurate results.


## Table of Contents

- [Overview](#Overview)
- [Key Findings](#key-results)
- [requiremnets](#requirements)
- [Set Up](#setup)
- [Contact](#contact)
- [Example Usage](#Example-Usage)

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
   git clone https://github.com/your-user/tree-delineation.git
   cd tree-delineation
```
2. Install reqirements
```py
    pip install numpy, laspy, open3d, sklearn
```
```R
    install.packages(c("lidR", "future", "RCSF", "terra", "tidyverse", "dbscan", "sf"))
```

## Example Usage
lorum ipsum



## Cite

If you use this code or methodology in your research, please cite:

> Lindbom, T. (2025). *An Evaluation of Machine Learning-Based Pine Tree Segmentation Techniques* University of Canterbury.

## Contact
Tim Lindbom - Linkedin: https://www.linkedin.com/in/tim-lindbom/ - email: lindbomtim@gmail.com
