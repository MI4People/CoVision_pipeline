"""
This file prepares the segmentation dataset before any training or testing. 
It must be run everytime we want a new train-val-test split and, after that, 
it is not needed to be ran again.
"""

import os
import sys
import random
import pandas as pd

def split_data(args, kit_id):
    """
    Build {kit_id}_filenames.csv file which contains the image names as indices and data_mode, namely
    train, val, or test mode, as column. Before doing so, it checks that all image and mask names coincide, 
    and they have the correct format. 
    
    The data mode is assigned randomly, preserving the train/val/split ratios
    specified in args.
    
    If filename already exists, it asks if we want to overwrite it.

    Args:
        args (Namespace): parameters
        kit_id (string): name of the test kit (eg. aconag, btnx, abbott, etc).
    """

    kit_folder = os.path.join(args.data_dir, kit_id)
    save_path = os.path.join(kit_folder, f'{kit_id}_filenames.csv')
    
    # Check if filename file already exists. If so, interrupt script
    if os.path.exists(save_path):
        overwrite = input(f"File {save_path} already exists! Do you want to overwrite it? (y/n): ")
        while overwrite not in ['y', 'n']:
            overwrite = input("Introduce 'y' or 'n': ")
        if overwrite == 'n':
            sys.exit('The file was not overwritten :)')
    
    # Store paths
    images_path = os.path.join(kit_folder, f'{kit_id}_images')
    masks_path = os.path.join(kit_folder, f'{kit_id}_masks')
    
    # Read filenames
    filenames_images = os.listdir(images_path) 
    filenames_masks = os.listdir(masks_path)

    # Make sure all images are .jpg and all masks .png
    if not all([f.endswith('.jpg') for f in filenames_images]):
        raise ValueError("Images must be in .jpg format!")
    if not all([f.endswith('.png') for f in filenames_masks]):
        raise ValueError("Masks must be in .png format!")
    
    # Read image and mask ids (filenames withouth extensions) and make sure they match
    ids_images = [f.replace('.jpg', '') for f in filenames_images]
    ids_masks = [f.replace('.png', '') for f in filenames_masks]
    assert set(ids_images) == set(ids_masks), "The number of images and masks do not coincide!"

    # Create list of randomly assigned train, val, and test labels, preserving the ratio specified in args
    n_train = int(len(ids_images)*args.train_test_split)
    n_test = len(ids_images) - n_train 
    n_train = int(n_train*args.train_val_split)
    n_val = len(ids_images) -  n_train - n_test

    data_mode = ['train']*n_train + ['test']*n_test + ['val']*n_val
    random.shuffle(data_mode)
    
    # Create dataframe with filenames as indices and (randomly assigned) data_mode as columns
    filenames_df = pd.DataFrame(data=data_mode, columns=['data_mode'], index=ids_images)
    # Store as csv file
    filenames_df.to_csv(os.path.join(kit_folder, f'{kit_id}_filenames.csv'))

    print(f"File {kit_id}_filenames.csv created succesfully!")



