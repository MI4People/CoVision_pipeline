""" 
This file contains the main class to define datasets for the classification model.
"""

import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from src.utils.miscellaneous import crop_zones

class MembraneZonesDataset(Dataset):
    def __init__(self, args, kit_id, data_mode='all', shots=None, transform=None):
        
        if data_mode not in ['all', 'train', 'val', 'test']:
            raise ValueError("data_mode must be 'all', 'train', 'val', or 'test'!")

        # Define attributes
        self.args = args
        self.kit_id = kit_id
        labels_dir = os.path.join(self.args.data_dir, kit_id, f'{kit_id}_labels.csv')
        membranes_dir = os.path.join(self.args.data_dir, kit_id, f'{kit_id}_membranes')
        self.data_mode = data_mode
        self.transform = transform
        self.n_zones = 2  # Only two-line kits

        # Load filenames of corresponding data_mode
        labels_df = pd.read_csv(labels_dir, index_col=0, dtype=str)
        if self.data_mode != 'all':
            labels_df = labels_df[labels_df.data_mode == self.data_mode]
        labels_df.drop(columns=['data_mode'], inplace=True)
            
        # If shots is specified, it picks a random subset from the dataframe
        if shots is not None:
            if shots > len(labels_df):
                raise ValueError(f"shots ({shots}) must be smaller than the number of elements in the dataset ({len(labels_df)})!")  
            labels_df = labels_df.sample(shots)
        
        # Store labels as list of lists (1 for red lines, 0 otherwise) and membrane paths from df indices
        self.labels = list(map(lambda l: [int(l[0]), int(l[1])], labels_df['line_sequence']))
        self.filenames = labels_df.index.tolist()
        self.membrane_paths = [os.path.join(membranes_dir, f'{fname}.jpg') for fname in self.filenames]
        
        print(f'There are {len(self)} membranes ({len(self)*self.n_zones} zones) in the {self.kit_id} kit for {self.data_mode} data mode')
            
    def __len__(self):
        return len(self.membrane_paths)

    def __getitem__(self, idx):

        # Get corresponding membrane image and label
        membrane = cv2.imread(self.membrane_paths[idx])
        labels = self.labels[idx]
        # Crop relevant zones
        zones = crop_zones(self.args, membrane)
        # Apply transforms if applicable
        if self.transform is not None:
            zones = self.transform(zones)
        else:  # Convert to Tensor (also change channel axis and scale to [0, 1])
            zones = torch.stack([F.to_tensor(z) for z in zones], dim=0)  # Handle fake "batch" direction
        labels = torch.as_tensor(labels, dtype=torch.float)
        
        return zones, labels

