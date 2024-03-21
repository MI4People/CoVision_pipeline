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

from utils_cla.miscellaneous import crop_zones

class MembraneZonesDataset(Dataset):
    def __init__(self, args, kit_id, data_mode='all', shots=None, transform=None):
        
        # Define attributes 
        self.args = args
        # Build list of kit_ids and shots
        if isinstance(kit_id, str):
            self.kit_id = [kit_id]
            if isinstance(shots, int) or shots is None:
                self.shots = [shots]
            else:
                raise ValueError("shots must be an interger or None")
        elif isinstance(kit_id, list):
            self.kit_id = kit_id
            if isinstance(shots, list) and len(shots)==len(self.kit_id):
                self.shots = shots
            else:
                raise ValueError("shots must be a list of integers or None values (same length as kit_id)")
        else:
            raise ValueError("kit_id must be an instance of str or list")

        if data_mode not in ['all', 'train', 'val', 'test']:
            raise ValueError("data_mode must be 'all', 'train', 'val', or 'test'!")
        self.data_mode = data_mode
        self.transform = transform
        self.n_zones = 2  # Only two-line kits

        # Build labels, filenames and membrane paths of all kits in kit_id list
        self.labels, self.filenames, self.membrane_paths = self.build_labels_and_membrane_paths()
        
        print(f'There are {len(self)} membranes ({len(self)*self.n_zones} zones) in the {"-".join(self.kit_id)} kit(s) for {self.data_mode} data mode')
            
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

    def build_labels_and_membrane_paths(self):
        """
        Build labels and membrane paths by reading the csv files from all kits in kit_id list. 
        It filters the examples to belong to data_mode and picks a subset of size shots (if not None).
        """

        # To store labels, filenames and membrane paths for all kits in kit_id list
        all_labels = []
        all_filenames = []
        all_membrane_paths = []

        for (i, _id) in enumerate(self.kit_id):
            # Read label and membrane directories
            labels_dir = os.path.join(self.args.data_dir, _id, f'{_id}_labels.csv')
            membranes_dir = os.path.join(self.args.data_dir, _id, f'{_id}_membranes')
            # Build dataframe
            labels_df = pd.read_csv(labels_dir, index_col=0, dtype=str)
            # Filter to specific data_mode
            if self.data_mode != 'all':
                labels_df = labels_df[labels_df.data_mode == self.data_mode]
            labels_df.drop(columns=['data_mode'], inplace=True)
            # If shots is specified, it picks a random subset from the dataframe
            shots_id = self.shots[i]
            if shots_id is not None:
                if shots_id > len(labels_df):
                    raise ValueError(f"shots ({shots_id}) must be smaller than the number of elements in the dataset ({len(labels_df)})!")  
                labels_df = labels_df.sample(shots_id)
            # Build final labels as list of lists (1 for red lines, 0 otherwise) and membrane paths from df indices
            labels = list(map(lambda l: [int(l[0]), int(l[1])], labels_df['line_sequence']))
            filenames = labels_df.index.tolist()
            membrane_paths = [os.path.join(membranes_dir, f'{fname}.jpg') for fname in filenames]
            # Update full lists
            all_labels.extend(labels)
            all_filenames.extend(filenames)
            all_membrane_paths.extend(membrane_paths)

        return all_labels, all_filenames, all_membrane_paths

