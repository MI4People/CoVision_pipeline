# README

This folder contains all documents that compose the full segmentation pipeline, from data preparation to inference. 

In this `README.md` file we describe briefly the content of each document and sub-folder. Some of these sub-folders contain specific `README.md` files to describe their functionalities in more detail.

* `config.yaml`: Configuration file containing all parameters and arguments used for data storage, pre-processing, and model training, testing, and inference. Such a file is passed as an argument to almost all other scripts within this folder.
* `data`: Folder containing the pre-processed data for training, and testing the segmentation model.
* `src`: Folder containing the main scripts for the general pipeline.
* `utils`: Folder containing secondary util files for different purposes, such as visualization.
* `models`: Folder where we store trained models.
* `results`: Folder containing some outcomes of performing inference in different datasets. They include `csv` files listing the IoU metrics obtained upon applying a segmentation model over a given set, as well as images showing the predictions (segmentation masks for kit and membrane).
* `notebooks`: Folder containing jupyter notebooks, used to ilustrate how to use different scripts. The `full_segmentation_pipeline.ipynb` notebook is the most important one. It demonstrates how to perform each one of the full-pipeline steps: data preparation, model training, model evaluation and membrane extraction.


## TODO:

* Add description of arguments in configuration file
* Add MLFlow to organize model training. 
* Document each function correctly (describe what they do with docstrings).
* Make save_model in TrainerSegmentation class also saves separately only the model's state (very light compared to full disctionary for resuming training).






