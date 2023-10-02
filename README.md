# Pyramidal 3D Feature Fusion on Polar-Grids for Fast and Robust Traversability Analysis on CPU

![Alt text](img/img.png)
![Alt text](img/grid_top_view.png)


## Requirements

 - yaml-cpp

    > refer to https://github.com/jbeder/yaml-cpp

 - Open3D (tested with Open3D==0.16 (Release))
    
    > refer to https://github.com/isl-org/Open3D/releases/tag/v0.16.0, BUT
    
    > follow these guidelines https://github.com/isl-org/Open3D/issues/2286#issuecomment-765353244

 - OpenCV >= 4.2 


## Build & Run

 - Fill config.yaml, in particular the paths to the dataset (and output path) <br>
 > mkdir -p build & cd build <br>
 > cmake .. <br>
 > make

 - in folder "models" we put some sample models (used in the paper)
      - they can be used to have an immediate feedback of how the system works

 > ./test

 ## Train models

 - First of all we need data, in particular features belonging to cells, to train our models
 - The following will produce the data (in multi-threading setting to speed up the process if frames are numerous)
 > ./produce_data

 - UP TO NOW: produce data only level by level

 - then, we can train our models 

 > ./train


## Cite Us
If you use this code in an academic context, please cite our [paper](https://www.sciencedirect.com/science/article/pii/S092188902300163X/pdfft?md5=757ff2721816cb1de2334e60069bc588&pid=1-s2.0-S092188902300163X-main.pdf):

@article{FUSARO2023104524,
title = {Pyramidal 3D feature fusion on polar grids for fast and robust traversability analysis on CPU},
author = {Daniel Fusaro and Emilio Olivastri and Ivano Donadi and Daniele Evangelista and Emanuele Menegatti and Alberto Pretto},
journal = {Robotics and Autonomous Systems},
volume = {170},
pages = {104524},
year = {2023},
issn = {0921-8890},
doi = {https://doi.org/10.1016/j.robot.2023.104524}
}