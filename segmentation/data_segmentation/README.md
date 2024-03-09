# README

This folder contains all raw data for training, and testing the segmentation mode. The data is (and must be) organized following specific conventions: each kit has its own folder, whose name is `kit_id`, in lowercase (e.g. `aconag`). Each of these folders contains two sub-folders with names `kit_id_images` and `kit_id_masks`. The former must contain images with .jpg extension and the latter their corresponding masks in .png format (green pixels for membrane, red for kit, blue for background). Each .jpg image must have one and only one .png mask with the same filename (e.g. `IMG_0254.jpg` (image) and `IMG_0254.png` (mask)). 

On top of these two sub-folders, the `kit_id` folders that are ready for training, and testing, must contain a `kit_id_filenames_csv` file. Such a file contains all image (and mask) filenames and their assigned `data_mode`, namely `train`, `val`, or `test`, dependeing whether they will be used for training, validation or testing, respectively. This file is created with the function `build_filenames_file`, inside the `utils_dataset.py` script (check its documentation for further details).


