# README

This folder contains all documents that compose the full segmentation pipeline, from data preparation to inference. 

In this `README.md` file we describe briefly the content of each document and sub-folder. Some of these sub-folders contain specific `README.md` files to describe their functionalities in more detail.

* `config_segmentation.yaml`: Configuration file containing all parameters and arguments used for data storage, pre-processing, and model training, testing, and inference. Such a file is passed as an argumennt to almost all other scripts within this folder.
* `data_segmentation`: Folder containing the raw data for training, and testing the segmentation model.
* `pipeline_segmentation`: Folder containing the main scripts for the general pipeline.
* `utils_segmentation`: Folder containing secondary util files for different purposes, such as data pre-processing, and visualization.
* `saved_models`: Folder where we store trained models and parameters.
* `results`: Folder containing some outcomes of performing inference in different datasets. They include `csv` files listing the IoU metrics obtained upon applying a segmentation model over a given set, as well as images showing the predictions (segmentation masks for kit and membrane).
* `notebooks_segmentation`: Folder containing jupyter notebooks, used to ilustrate how to use different scripts. The `full_segmentation_pipeline.ipynb` notebook is the most important one. It demonstrates how to perform each one of the full-pipeline steps: data preparation, model training, model testing and membrane extraction.


## TODO:

* Add description of arguments in configuration file
* Organize trained models inside `saved_models` in a way that it is clear how each model was trained (which dataset was used, which (hyper)parameters, etc) and which performance it achieved. Maybe a csv file containg each model as a row, and each property (dataset, learning rate, etc) as a column?
* Document each function correctly (describe what they do with docstrings).





