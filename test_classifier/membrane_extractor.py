"""
This file contains functions to extract the membrane region of a given raw image, which will then be sent to a classifier.
"""

import argparse
import math
import json
import numpy as np
import torch
from torchvision.transforms import functional as F
import cv2

# Custom packages
from segmentation import transformations_seg, utils_seg

from transformations_seg import resize_image
from utils_seg.miscellaneous import compute_bounding_box_coordinates

def extract_corrected_membrane(args, image_path, model, return_mask=False):
    """
    Extract the corrected membrane region from a single raw image
    """
    if model.training:
        model.eval()

    # Split configuration file
    data_args = argparse.Namespace(**args.data_args)
    inference_args = argparse.Namespace(**args.inference_args)

    # Read image
    image = cv2.imread(image_path)
    # Resize image 
    image_r = resize_image(image, data_args.resize_height)
    # Convert to Tensor
    image_t = F.to_tensor(image_r)
    
    # Run inference
    (mask_kit, mask_membrane), (box_kit, box_membrane), (score_kit, score_membrane) = run_inference(data_args, inference_args, image_t, model)

    # Compute coordinates of the minimal rectangle enclosing the membrane mask
    rect_membrane = compute_rectangle(mask_membrane)
    # Estimate rotation angle
    angle = compute_angle(rect_membrane)
    print(f'Rotated angle: {angle}')
    
    # Get kit bounding box coordinates
    x1, y1, x2, y2 = box_kit
    
    # Scale box coordinates using ratio between orginal and resized images
    y_scale, x_scale = [image.shape[i]/image_r.shape[i] for i in [0, 1]]
    x1_s, x2_s = int(x1 * x_scale), int(x2 * x_scale)
    y1_s, y2_s = int(y1 * y_scale), int(y2 * y_scale)
        
    # Resize membrane mask to be of the original image size (upsample)
    mask_membrane_big = resize_image(mask_membrane, new_height=image.shape[0])
    # Crop original image and membrane mask to contain only the kit, using the scaled box coordinates
    image_crop = image[y1_s:y2_s, x1_s:x2_s]
    mask_membrane_crop = mask_membrane_big[y1_s:y2_s, x1_s:x2_s]

    # Rotate croped image and mask to leave them vertical
    image_rot = rotate_image(image_crop, -angle)
    mask_membrane_rot = rotate_image(mask_membrane_crop, -angle)
    
    # Build bounding box for the rotated mask and use its coordinates to crop the rotated image
    x1, y1, x2, y2 = compute_bounding_box_coordinates(mask_membrane_rot)
    membrane_correct = image_rot[y1:y2, x1:x2]
    print(f"Kit, Membrane scores: ({score_kit:.3f}, {score_membrane:.3f})")

    if return_mask:
        mask_membrane_correct = mask_membrane_rot[y1:y2, x1:x2]
        return membrane_correct, mask_membrane_correct, score_kit, score_membrane
    else:
        return membrane_correct, score_kit, score_membrane

def run_inference(data_args, inference_args, image, model):
    """
    Run inference on single image an returns scores, boxes and masks for each class
    """

    # Exclude background class
    classes = data_args.classes[1:]
    class_ids = data_args.class_ids[1:]

    # Inference step
    with torch.no_grad():
        predictions = model(image.unsqueeze(0))[0]  # Add batch axis -> forward pass -> remove batch axis

    # Store scores, boxes and masks
    scores = np.zeros(2)
    boxes = np.zeros([2, 4])
    masks = np.zeros([2, *image.shape[1:]], dtype=np.uint8)

    labels = predictions['labels'].tolist()
    # Loop over classes (i.e. kit and membrane)
    for i, cls in enumerate(classes):
        if class_ids[i] in labels:  # Check whether there is at least one prediction for that class

            # Get the maximum confidence class location (i.e. first occurrence in list)
            class_loc = labels.index(class_ids[i])

            # Get best class score
            class_score = predictions['scores'].tolist()[class_loc]

            # Check score is above pre-stablished threshold
            if class_score < inference_args.score_thresholds[i]:
                raise ValueError(f"Score {class_score:.3f} below threshold {inference_args.score_thresholds[i]:.3f} for {cls} class!")
            
            # Get best class boxes and masks
            class_box = predictions['boxes'][class_loc]
            class_mask = predictions['masks'][class_loc, 0]
            # Binarize masks
            class_mask = (class_mask >= inference_args.mask_thresholds[i]).to(torch.uint8)

            # Update values
            scores[i] = class_score
            boxes[i] = class_box.to('cpu')
            masks[i] = class_mask.to('cpu')

        else:  # If there is no prediction, we leave all zeros in the dictionary
            raise ValuError(f'{cls} is missing from the prediction!')

    return masks, boxes, scores

def compute_rectangle(mask):
    """
    Function to compute the coordinates of the minimal (rotated) rectangle enclosing the mask.
    
    Args:
        mask (np.narray): image mask with shape (H, W) and with values [0, 1]
    
    Return:
        rect_coords (np.array): coordinates of rectangle [[xlb, ylb], [xrb, yrb], [xrt, yrt], [xlt, ylt]]
    """

    # # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Select only one contour, the one with maximum area
    contour = max(contours, key=cv2.contourArea)
    # Build minimal rectangular box containing the contour (it also returns angle...)
    rect = cv2.minAreaRect(contour)
    # Get coordinates of rectangle in a better format: list of 4 vertex points
    box_points = cv2.boxPoints(rect).astype('int')
    
    def sort_points(points):
        """
        Make sure the box points are always sorted in the same way
        """
    
        # Sort the points in box based on their y-coordinates
        points_ysorted = box_points[np.argsort(points[:, 1])]

        # Grab the bottommost and topmost points from the sorted y-coordinate points (NOTICE: (0, 0) is the topleft corner)
        topmost, bottommost  = points_ysorted[:2, :], points_ysorted[2:, :]

        # Sort the topmost coordinates according to their x-coordinates
        lefttop, righttop = topmost[np.argsort(topmost[:, 0]), :]

        # Sort the bottommost coordinates according to their x-coordinates
        leftbottom, rightbottom = bottommost[np.argsort(bottommost[:, 0]), :]
        
        return np.array([leftbottom, rightbottom, righttop,  lefttop], dtype='int')

    rect_coords = sort_points(box_points)
    
    return rect_coords

def compute_angle(rect_coords):
    """
    Compute angle of rotation of a given rectangle
    
    Args:
        rect_coords (np.ndarray): Assumes format [[xlb, ylb], [xrb, yrb], [xrt, yrt], [xlt, ylt]]
        
    Return:
        angle (int): angle of rotation.
    """
    
    # Decompose coordinates
    (xlb, ylb), (xrb, yrb), (xrt, yrt), (xlt, ylt) = rect_coords
    # Compute left and right angles
    left_angle = math.atan((xlb - xlt) / (ylb - ylt)) * (180 / math.pi)
    right_angle = math.atan((xrb - xrt) / (yrb - yrt)) * (180 / math.pi)
    # Calculate average angle
    angle = int(round((left_angle + right_angle) / 2))
                            
    return angle

def rotate_image(img, angle):
    """
    Function to rotate image in a specified (integer) angle without cropping any part of the image
    """
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    # Create rotation matrix and rotate all size to get new size
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    # Update center
    M[:,-1] += (size_new - size_reverse) / 2.
    # Rotate image
    img_rotated = cv2.warpAffine(img, M, tuple(size_new.astype(int)))

    return img_rotated

