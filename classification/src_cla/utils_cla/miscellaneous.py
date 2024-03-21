"""
File containing various functions used along the way in the full classification pipeline.
"""

import numpy as np
import random
import torch
import cv2

def set_seed(seed):
    """
    Set seed for numpy, torch, and all other random processes.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"Random seed set as {seed}")


def crop_zones(args, membrane):
    """
    Crop membrane into the relevant zones using universal cropping locations stored in args
    Return:
        zones (np.ndarray): The cropped zones concatenated into an array (e.g. shape (2, 160, 160, 3))
    """ 
    # Resize image to standard shape
    membrane_r = cv2.resize(membrane, args.membrane_shape, interpolation=cv2.INTER_NEAREST)
    # Scale cropping locations to membrane's height
    crop_locs_scaled = [int(loc*args.membrane_shape[1]) for loc in args.crop_locs]
    
    # Store cropped zones
    zones = []
    for i in range(len(crop_locs_scaled) - 1):
        # Read zone, resize and add extra direction for latter concatenation
        zone = cv2.resize(membrane_r[crop_locs_scaled[i]:crop_locs_scaled[i+1]], args.zone_shape, interpolation=cv2.INTER_NEAREST)
        zones.append(np.expand_dims(zone, 0))
    zones = np.concatenate(zones, axis=0)

    return zones
