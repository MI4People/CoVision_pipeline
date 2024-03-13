""" Dataloader for all datasets. """
import os
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class FullMembraneDataset(Dataset):
    def __init__(self, args, set_mode, transform=None):
        
        self.args = args
        self.set_mode = set_mode
        self.transform = transform

        # Membrane paths
        membranes_folder = os.path.join(self.args.dataset_dir, self.args.kit_id, self.set_mode)
        membrane_filenames = os.listdir(membranes_folder)
        self.membrane_paths = [os.path.join(membranes_folder, fname) for fname in membrane_filenames]

        # Labels
        label_file = pd.read_excel(os.path.join(self.args.dataset_dir, self.args.kit_id, 'labels.xlsx'), index_col='Sample ID')
        self.labels = [label_file.loc[fname.replace('.jpg', '')].tolist() for fname in membrane_filenames]

        # Map original labels in format [1, 1, 1] to binary, 1 or 0
        self.labels = self.labels_to_binary(self.labels)

        print('There are {} images in the {} set for {} mode'.format(len(self.membrane_paths), self.args.kit_id, self.set_mode))
            
    def __len__(self):
        return len(self.membrane_paths)

    def __getitem__(self, idx):

        # Get corresponding membrane image (PIL) and label
        image = Image.open(self.membrane_paths[idx])
        label = self.labels[idx]

        # Apply transforms if applicable
        if self.transform is not None:
            image_t = self.transform(image)
        else:  # Convert to Tensor
            image_t = F.to_tensor(image)  # Also scales to [0, 1] range and brings channel to first position!
        
        # Float and dimension expansion required for criterion later
        label_t = torch.as_tensor(label, dtype=torch.float).unsqueeze(dim=0)
        
        return image_t, label_t
    
    @staticmethod
    def labels_to_binary(labels):
        diagnosis_mapping = {"[1, 0, 0]": 0, "[1, 0, 1]": 1, "[1, 1, 0]": 1, "[1, 1, 1]": 1}
        labels_binary = [diagnosis_mapping[str(label)] for label in labels]

        return labels_binary
        

if __name__ == '__main__':
    pass