"""
This file contains the pre-processing and data agumentation transformations for the classification dataset.
"""

import numpy as np
import torch
from torchvision import transforms
import cv2

class PreprocessMembraneZones:
    def __init__(self, w=224, h=224):
        
        # Statistics from ImageNet training dataset (for pretrained models)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.w, self.h = w, h

        # Common resizing transformation
        self.resize = transforms.Compose([transforms.ToPILImage(), transforms.Resize((self.h, self.w))])
        # For data augmentation
        self.augment = None
        # Final common transformation
        self.final_transformation = transforms.Compose([transforms.ToTensor(), 
                                                        transforms.Normalize(self.mean, self.std)])
        
        self.full_transformation = self.build_full_transformation()
        
            
    def __call__(self, membrane_image):
        
        membrane_image = torch.stack([self.full_transformation(zone) for zone in membrane_image], dim=0)

        return membrane_image
    
    def build_full_transformation(self):
        if self.augment is not None:
            return transforms.Compose([self.resize, self.augment, self.final_transformation])
        else:
            return transforms.Compose([self.resize, self.final_transformation])
        

class AugmentedMembraneZones(PreprocessMembraneZones):
    def __init__(self, w=248, h=248):
        super().__init__(w=w, h=h)
        self.augment = transforms.Compose([
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                    transforms.GaussianBlur(kernel_size=(1, 9), sigma=(0.1, 5.))])
        
        self.full_transformation = self.build_full_transformation()

def resize_image(image, new_height):
    """
    Resize image to a a new height, preserving the original width/height ratio.
    """
    ratio = image.shape[1] / image.shape[0]
    new_width = int(new_height*ratio)
    image = cv2.resize(image, (new_width, new_height))
    return image