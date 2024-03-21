# README

This folder contains all documents that compose the full classification pipeline, from data preparation to evaluation. 

In this `README.md` file we describe briefly the content of each document and sub-folder. Some of these sub-folders contain specific `README.md` files to describe their functionalities in more detail.

* `config_cla.yaml`: Configuration file containing several parameters and arguments used for data storage, pre-processing, etc. Such a file is passed as an argument to almost all other scripts within this folder.
* `data_cla`: Folder containing the pre-processed datasets as it must be fed into the training and evaluation pipelines. Check out the `README.md` file therein for more details.
* `src_cla`: Folder containing the main scripts for the general pipeline.
* `models_cla`: Folder where we store trained models and their parameters. Use when running experiments in development stage as well as to store registered models in staging or deployment stages. Check out the `README.md` file therein for more details.
* `notebooks_cla`: Folder containing jupyter notebooks, used for the pre-processing pipeline as well as for running experiments.







