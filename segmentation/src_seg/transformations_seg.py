"""
File containing data transformations to use during training and inference.
"""

import cv2
import albumentations as A


class TransformationSegmentationTraining():
    """
    Transformation for segmentation training dataset.
    """
    def __init__(self, args):
        self.args = args

        # Rotate
        self.rotate = A.Rotate(limit=args.rotate_limit, p=args.rotate_p, border_mode=cv2.BORDER_CONSTANT)
        # Horizontal flip
        self.horizontal_flip = A.HorizontalFlip(p=args.horizontal_flip_p)
        # Affine
        self.affine = A.Affine(scale=args.affine_scale, translate_percent=args.affine_translate_percent, p=args.affine_p)
        # Perspective
        self.perspective = A.Perspective(scale=args.perspective_scale, p=args.perspective_p)      
        # Blur
        self.blur = A.Blur(blur_limit=args.blur_limit, p=args.blur_p)
        # Color jitter
        self.color_jitter = A.ColorJitter(brightness=args.color_jitter_brightness, 
                                        contrast=args.color_jitter_contrast,
                                        saturation=args.color_jitter_saturation, 
                                        p=args.color_jitter_p)
        # Downscale
        self.downscale = A.Downscale(scale_min=args.downscale_min, scale_max=args.downscale_max,
                                     interpolation=cv2.INTER_LINEAR, p=args.downscale_p)
        # GaussNoise
        self.gauss_noise = A.GaussNoise(var_limit=args.gauss_noise_var_limit, p=args.gauss_noise_p)
        
        # Composite transformation
        self.transform = A.Compose([self.rotate, 
                                    self.horizontal_flip,
                                    self.affine,
                                    self.perspective,
                                    A.OneOf([
                                        self.blur, 
                                        self.color_jitter], p=1.0),
                                    self.downscale,
                                    self.gauss_noise])
                                                                

    def __call__(self, image, mask):
        
        transform_dict = self.transform(image=image, mask=mask)

        return transform_dict['image'], transform_dict['mask']


def resize_image(image, new_height):
    """
    Resize image to a a new height, preserving the original width/height ratio.
    """
    ratio = image.shape[1] / image.shape[0]
    new_width = int(new_height*ratio)
    image = cv2.resize(image, (new_width, new_height))
    return image