"""
TODO:
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from transformations_cla import resize_image

def split_data(args, kit_id):
    """
    Perform a random train/val/test split preserving the distribution of the entire dataset. 
    This means that we preserve the train/val/test ratio for each line sequence. 
    The split is incorporated as a data_mode column in the already present {kit_id}_labels.csv.
    If a split was already performed before, the function asks if we want to overwrite the file.
    """
    
    # Read {kit_id}_labels.csv file
    labels_dir = os.path.join(args.data_dir, kit_id, f'{kit_id}_labels.csv')
    labels_df = pd.read_csv(labels_dir, index_col=0, dtype=str)
    
    # Check if data_mode column already present. If so, ask whether to overwrite it.
    if 'data_mode' in labels_df.columns:
        overwrite = input(f"File {labels_dir} already has a train/val/test split! Do you want to overwrite it? (y/n): ")
        while overwrite not in ['y', 'n']:
            overwrite = input(f"Introduce 'y' or 'n': ")
        if overwrite == 'n':
            sys.exit('The file was not overwritten :)')

    sequence_counts = labels_df.value_counts('line_sequence').items()
    for seq, count in sequence_counts:

        n_train = int(count*args.train_test_split)
        n_test = count - n_train 
        n_train = int(n_train*args.train_val_split)

        # Get filenames with the specific sequence and apply random shuffle
        image_filenames = labels_df[labels_df['line_sequence'] == seq].index.to_list()
        np.random.shuffle(image_filenames)

        # Assign test, train, and val labels
        labels_df.loc[image_filenames[:n_test], 'data_mode'] = 'test'
        labels_df.loc[image_filenames[n_test:n_test + n_train], 'data_mode'] = 'train'
        labels_df.loc[image_filenames[n_test + n_train:], 'data_mode'] = 'val'
    
    # Store as csv file
    labels_df.to_csv(labels_dir)

    print(f"File {labels_dir} created succesfully!")


def check_cropped_zones(args, kit_id):
    """
    Display membrane images together with the universal cropping locations as green lines. 
    This function is used to check whether the membranes zones are been cropped correctly. 
    This is a mandatory check we need to run for every new test kit. So far it only works for 2-line tests.
    """

    # Load membrane full paths
    membranes_dir = os.path.join(args.data_dir, kit_id, f'{kit_id}_membranes')
    membrane_paths = [os.path.join(membranes_dir, img) for img in os.listdir(membranes_dir)]

    # Number of images to display, row and columns
    n_imgs = len(membrane_paths)
    n_cols = min(n_imgs, 20)
    n_rows = (n_imgs//n_cols + 1) if n_imgs % n_cols else n_imgs//n_cols

    height = 480  # New height for resizing, common to all membranes

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, 4*n_rows))

    for i in range(n_rows):
        for j in range(n_cols):
            # Catch single row case
            if n_rows > 1:
                ax = axs[i, j]
            else:
                ax = axs[j]
            if n_cols*i+j < n_imgs:
                # Read and resize membrane image
                membrane = resize_image(cv2.imread(membrane_paths[n_cols*i+j]), height)

                # Draw green horizontal lines at each cropping location to devide zones
                x1, x2 = 0, membrane.shape[1]
                for loc in args.crop_locs:
                    y = int(loc*height)
                    cv2.line(membrane, (x1, y), (x2, y), (0, 255, 0), thickness=2)

                ax.axis('off')
                ax.imshow(membrane[:,:,::-1])
            else:
                ax.axis('off')

    plt.suptitle(f'{kit_id} ({n_imgs})', size='xx-large')
    fig.tight_layout()
    plt.show()