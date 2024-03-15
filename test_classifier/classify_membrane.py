"""
TODO:
"""

import argparse
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np

# Custom packages
from classification import transformations_cla, utils_cla

from utils_cla.miscellaneous import crop_zones
from transformations_cla import PreprocessMembraneZones

def classify_membrane(args, membrane, model, show_bool=False):
    """
    Takes a raw image of a membrane and log whether it corresponds to a positive, negative or invalid LFA test.
    """

    if model.training:
        model.eval()
    
    # Split configuration file
    data_args = argparse.Namespace(**args.data_args)
    inference_args = argparse.Namespace(**args.inference_args)
    # Crop relevant zones
    zones = crop_zones(data_args, membrane)
    # Apply transformation
    transformation = PreprocessMembraneZones()
    zones = transformation(zones)
    # Run inference
    with torch.no_grad():
        y_pred = model(zones)
    # Binarize prediction
    binary_pred = (y_pred > 0.5).squeeze().to(int).tolist()
    # To string
    pred_str = str(binary_pred[0]) + str(binary_pred[1])
    # Map to diagnosis
    result = inference_args.diagnosis_map[pred_str]
    if result == 1:
        result_msg = 'Positive'
    elif result == 0:
        result_msg = 'Negative'
    elif result == 99:
        result_msg = 'Invalid'

    print(f'{result_msg} ({pred_str})')

    if show_bool:
        plt.imshow(membrane[:,:,::-1])
        plt.title(f'{result_msg} ({pred_str})')
        plt.axis('off')
        plt.show()
    

