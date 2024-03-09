"""
File containing the TesterSegmentation class, the main class for testing a segmentation model.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange  # Just to log fancy training progress bar

# Custom packages
from utils_segmentation.utils import collate_fn, compute_iou_mask, compute_iou_box
from utils_segmentation.visualization import show_images
from pipeline_segmentation.dataset_segmentation import LFASegmentationDataset


class TesterSegmentation():
    def __init__(self, args, model, device=None, n_batches=2, n_workers=1):
        
        self.args = args
        self.model = model
        self.device = device
        self.n_batches = n_batches
        self.n_workers = n_workers
        
        # Split configuration file
        self.data_args = argparse.Namespace(**args.data_args)
        self.inference_args = argparse.Namespace(**args.inference_args)
        
        # Classes excluding background
        self.classes = self.data_args.classes[1:]
        self.class_ids = self.data_args.class_ids[1:]
        self.n_classes = len(self.classes)

        # Align input device with model's device
        if self.device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.model.to(device)
        # Set model in evaluation mode
        self.model.eval()
    
    def test(self, kit_id, data_mode='test', show_bool=False, save_bool=False, save_filename=None):
        """
        Main function. Run inference on all images in the corresponding set 
        and save all scores and IoU in a csv file.
        """

        if data_mode not in ['train', 'test', 'val']:
            raise ValueError("data_mode must be 'train', 'val', or 'test'!")

        # Initiialize data loader
        loader = self.init_dataloader(kit_id, data_mode)

        # Run inference on all data, returning predictions and metrics
        predictions, metrics = self.run_inference(loader, data_mode)
        
        # Show all images and their predictions
        if show_bool:
            images = [img.to('cpu') for img, _ in loader.dataset]
            show_images(images, predictions, metrics)

        # Format metrics as Dataframe
        metrics_df = self.get_metrics(metrics, image_names=loader.dataset.filenames)

        # Save metrics to csv file
        if save_bool:
            if save_filename is not None:
                save_path = os.path.join(self.inference_args.output_dir, f'{kit_id}_{data_mode}_{save_filename}_results.csv')
            else:
                save_path = os.path.join(self.inference_args.output_dir, f'{kit_id}_{data_mode}_results.csv')

            # Check if filename file already exists.
            if os.path.exists(save_path):
                overwrite = input(f"File {save_path} already exists! Do you want to overwrite it? (y/n): ")
                while overwrite not in ['y', 'n']:
                    overwrite = input(f"Introduce 'y' or 'n': ")        
                if overwrite == 'n':
                    print('The file was not overwritten :)')
                if overwrite == 'y':
                    metrics_df.to_csv(save_path)
            else:
                metrics_df.to_csv(save_path)

        # Log mean value of the metrics
        print('Mean metrics')
        for k, v in metrics_df.mean().items():
            print(f'{k}: {v:.4f}')
            
        return metrics_df
    
    def run_inference(self, loader, data_mode):
        """
        Make predictions for all batches in loader.
        """

        # List of dictionaries for predictions and metrics for the whole dataset in loader.
        predictions_list = []
        metrics_list = []

        # To log fancy progress bar
        loop = tqdm(loader)
        loop.set_description(f"Testing on {data_mode} set")

        for images, targets in loop:

            # Send images and targets to same device as model
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Inference step over single batch
            predictions_list_batch, metrics_list_batch = self.run_inference_batch(images, targets)

            # Update full output
            predictions_list += predictions_list_batch
            metrics_list += metrics_list_batch

            # Update progress bar
            loop.update()

        return predictions_list, metrics_list

    def run_inference_batch(self, images, targets):
        """
        Make predictions for each batch of images, and select the masks, and boxes with the best scores for each class 
        and image. The predicted quantities are then compared with the targets to extract the mask and box IoU 
        for each class.

        Returns:
            predictions_list: list of dictionaries of the form {masks, boxes} for each image
            metrics_list: list of dictionaries of the form {scores, iou_masks, iou_boxes} for each image.
        """

        # List of dictionaries for predictions and metrics for the whole batch.
        predictions_list = []
        metrics_list = []

        # Inference step
        with torch.no_grad():
            predictions = self.model(images)

        for target, pred in zip(targets, predictions):

            # Get labels and scores, which are aligned
            labels = pred['labels'].tolist()
            scores = pred['scores'].tolist()
            assert len(labels) == len(scores)

            # Get boxes and masks, which are also naturally aligned
            boxes = pred['boxes']
            masks = pred['masks']
            assert boxes.shape[0] == masks.shape[0] == len(labels)

            # Keep track of best-score mask and boxes predictions for each image
            pred_dict = {'masks': torch.zeros_like(target['masks']),
                        'boxes': torch.zeros_like(target['boxes'])}

            # Keep track of best scores, and average IoUs for each image
            metrics_dict = {'scores': np.zeros(self.n_classes),
                            'iou_masks': np.zeros(self.n_classes),
                            'iou_boxes': np.zeros(self.n_classes)}

            # Loop over classes (i.e. kit and membrane)
            for i, cls in enumerate(self.classes):
                if self.class_ids[i] in labels:  # Check whether there is at least one prediction for that class

                    # Get the maximum confidence class location (i.e. first occurrence in list)
                    class_loc = labels.index(self.class_ids[i])

                    # Get best class score
                    class_score = scores[class_loc]

                    # Get best class boxes and masks
                    class_box = boxes[class_loc]
                    class_mask = masks[class_loc, 0]

                    # Binarize masks
                    class_mask = (class_mask >= self.inference_args.mask_thresholds[i]).to(torch.uint8)

                    # Compute IoU
                    class_iou_mask = compute_iou_mask(class_mask, target['masks'][i])
                    class_iou_box = compute_iou_box(class_box, target['boxes'][i])

                    # Update dictionaries
                    pred_dict['masks'][i] = class_mask
                    pred_dict['boxes'][i] = class_box

                    metrics_dict['scores'][i] = class_score
                    metrics_dict['iou_masks'][i] = class_iou_mask
                    metrics_dict['iou_boxes'][i] = class_iou_box

                else:  # If there is no prediction, we leave all zeros in the dictionary
                    print(f'{cls} is missing from the prediction!')

            # Send mask and boxes to cpu
            pred_dict['masks'] = pred_dict['masks'].to('cpu')
            pred_dict['boxes'] = pred_dict['boxes'].to('cpu')

            # Update lists
            predictions_list.append(pred_dict)
            metrics_list.append(metrics_dict)

        return predictions_list, metrics_list


    def init_dataloader(self, kit_id, data_mode):
        """
        Initialize dataset and dataloader
        """

        dataset = LFASegmentationDataset(self.data_args,
                                         kit_id=kit_id,
                                         data_mode=data_mode,
                                         shots=None,
                                         transforms=None)

        loader = DataLoader(dataset=dataset,
                            batch_size=self.n_batches,
                            shuffle=False,
                            num_workers=self.n_workers,
                            collate_fn=collate_fn,
                            pin_memory=True)

        return loader

    @staticmethod
    def get_metrics(predictions, image_names):
        """
        Return metrics in the form of a Panda's dataframe with image_names as indices.
        """

        metrics_dict = {}

        for key in ['scores', 'iou_masks', 'iou_boxes']:
            for i, cls in enumerate(['kit', 'membrane']):
                metrics_dict[f'{key}_{cls}'] = [pred[key][i] for pred in predictions]

        metrics_df = pd.DataFrame.from_dict(metrics_dict)
        metrics_df.index = pd.Index(image_names, name='image_names')

        return metrics_df
