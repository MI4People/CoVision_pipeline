"""
File containg the main Dataset class for image segmentation. 
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import cv2
from torchvision.transforms import functional as F

# Custom packages
from utils_seg.miscellaneous import compute_bounding_box_coordinates
from transformations_seg import resize_image

class LFASegmentationDataset:
    def __init__(self, args, kit_id, data_mode='all', shots=None, transforms=None):
        
        if data_mode not in ['all', 'train', 'val', 'test']:
            raise ValueError("data_mode must be 'all', 'train', 'val', or 'test'!")

        # Configuration file and relevant features
        self.args = args
        self.kit_id = kit_id
        kit_folder = os.path.join(self.args.data_dir, kit_id)
        self.data_mode = data_mode
        
        # Transformations
        self.transforms = transforms
        
        # Load filenames of corresponding data_mode
        filenames_df = pd.read_csv(os.path.join(kit_folder, f'{kit_id}_filenames.csv'), index_col=0)
        if self.data_mode == 'all':
            self.filenames = filenames_df.index.to_list()
        else:
            self.filenames = filenames_df[filenames_df.data_mode == self.data_mode].index.to_list()
            
        # If shots is specified, it picks a random subset from filenames
        if shots is not None:
            if shots > len(self.filenames):
                raise ValueError(f"shots ({shots}) must be smaller than the number of elements in the dataset ({len(self.filenames)})!")
                 
            random.shuffle(self.filenames)
            self.filenames = self.filenames[:shots]

        # Load image, and mask full filepaths
        self.image_paths = [os.path.join(kit_folder, f'{kit_id}_images', f'{n}.jpg') for n in self.filenames]
        self.mask_paths = [os.path.join(kit_folder, f'{kit_id}_masks', f'{n}.png') for n in self.filenames]
        
        print(f'Loaded {len(self.filenames)} image and mask paths for {self.kit_id} test kit in {self.data_mode} data mode')

    def __len__(self):
        
        return len(self.filenames)
    
    def __getitem__(self, idx):
        
        # Get corresponding image and mask path
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read image and mask as NumPy arrays
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        
        # Resize image (excluded from transformations because it is mandatory for efficiency)
        image = resize_image(image, self.args.resize_height)
        mask = resize_image(mask, self.args.resize_height)

        # Check that image and masks have the same dimensions    
        assert image.shape[:2] == mask.shape[:2], "Image and Masks have different dimensions!"
        
        # Apply transforms if applicable
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        
        # Build category mask, bounding boxes, and labels from RGB mask
        masks_cat, boxes, labels =  self.build_target_from_mask(mask)

        # Convert everything to a torch.Tensor
        image_t = F.to_tensor(image)  # Also scales to [0, 1] range and brings channel to first position!
        masks_t = torch.as_tensor(masks_cat, dtype=torch.uint8)
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        # Build target with ground-truths and image information
        target = {'masks': masks_t, 
                  'boxes': boxes_t,
                  'labels': labels_t}
        
        return image_t, target
    
    def build_target_from_mask(self, mask):
        """
        Create category mask, bounding box, and labels from a given RGB mask.

        Args:
            mask (array (H, W, 3)): mask from where all other variables are computed

        Return:
            masks (array (2, H, W) [0-1]): binary masks of the kit and membrane
            boxes (array (2, 4)): coordinates of the bounding boxes in [xmin, ymin, xmax, ymax] format
            labels (array [2, 1]): labels for each class (1 for kit, 2 for membrane)
        """

        height, width = mask.shape[0:2]
        masks_cat = np.zeros([2, height, width])
        boxes = np.zeros([2, 4])
        labels = np.array(self.args.class_ids[1:])  # For kit and membrane (exclude background)

        # Build category mask from RGB mask
        masks_cat[1] = np.all(mask == self.args.class_colors[2], axis=-1).astype(int)  # Membrane
        # Include membrane mask to kit mask because, apparently, segmentation task is easier
        masks_cat[0] = np.all(mask == self.args.class_colors[1], axis=-1).astype(int) + masks_cat[1] # Kit (+ Membrane)

        # Compute bounding box coordinates [xmin, ymin, xmax, ymax]
        boxes[0] = compute_bounding_box_coordinates(masks_cat[0])
        boxes[1] = compute_bounding_box_coordinates(masks_cat[1])        

        return masks_cat, boxes, labels
