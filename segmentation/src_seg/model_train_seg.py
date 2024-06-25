"""
File containing the TrainerSegmentation class, the main class for training and validating a segmentation model.
"""

import argparse
import copy
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam, lr_scheduler

from time import time
from datetime import datetime as dt
from datetime import timedelta
from tqdm.notebook import tqdm  # Just to log fancy training progress bar

# Custom packages
from utils_seg.miscellaneous import set_seed, compute_iou_mask, compute_iou_box
from transformations_seg import TransformationSegmentationTraining
from dataloader_seg import init_dataloader
from model_seg import get_segmentation_model


class TrainerSegmentation:
    def __init__(self, args, parameters, do_validation=True):

        # Assign attributes
        self.args = args
        self.parameters = argparse.Namespace(**parameters)
        self.do_validation = do_validation
        self.kit_id = self.parameters.kit_id
        self.val_set = self.parameters.val_set
        self.shots = self.parameters.shots

        # Split configuration file
        self.data_args = argparse.Namespace(**args.data_args)
        self.transformation_args = argparse.Namespace(**args.transformation_args)
        self.evaluation_args = argparse.Namespace(**args.evaluation_args)
        
        # Classes excluding background
        self.classes = self.data_args.classes[1:]
        self.class_ids = self.data_args.class_ids[1:]
        self.n_classes = len(self.classes)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters.device = self.device
        print(f'Using {self.device} device')

        # Metrics
        self.metrics_train = {'loss_classifier': [],
                              'loss_box_reg': [],
                              'loss_mask': [],
                              'loss_objectness': [],
                              'loss_rpn_box_reg': [],
                              'total_loss': []}
        if self.do_validation:
            self.metrics_val = {'score_kit': [],
                                'score_membrane': [],
                                'iou_mask_kit': [],
                                'iou_mask_membrane': [],
                                'iou_box_kit': [],
                                'iou_box_membrane': []}
        else:
            self.metrics_val = None

        # Set seed for reproducibility
        set_seed(self.parameters.seed)
        
        # Transformations
        self.transformation_train = TransformationSegmentationTraining(self.transformation_args)
        self.transformation_val = None

        # Load training and validation datasets and dataloaders
        self.loader_train, self.loader_val = self.init_dataloaders()
        self.n_train = len(self.loader_train)
        self.n_val = len(self.loader_val)

        # Model architecture, optimizer and scheduler
        self.model = get_segmentation_model(num_classes=len(self.data_args.classes),
                                        hidden_size=self.parameters.hidden_size).to(self.device)
        self.best_model = copy.deepcopy(self.model)  # Keep track of best model

        self.optimizer = Adam(params=self.model.parameters(), lr=float(self.parameters.lr))

        if self.parameters.scheduler_step is not None:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, 
                                                step_size=int(self.parameters.scheduler_step), 
                                                gamma=float(self.parameters.scheduler_gamma))
        else:
            self.scheduler = None

        # Timestamp to identify training runs
        time_str = dt.now().strftime('%Y-%m-%d_%H.%M.%S')  
        self.file_path = f'{self.data_args.save_path}/{time_str}'
        
        # To keep track best models and continue training from checkpoint
        self.best_iou_mean = 0.0
        self.metrics_of_best = {'score_kit': 0.0,
                                'score_membrane': 0.0,
                                'iou_mask_kit': 0.0,
                                'iou_mask_membrane': 0.0,
                                'iou_box_kit': 0.0,
                                'iou_box_membrane': 0.0}
        self.best_epoch = 0
        self.start_epoch = 0
        self.elapsed_time = timedelta(0)

    def train(self, save_state_bool=False, save_model_bool=False):
        """
        Train and validate the model for all epochs, store the training and
        validation metrics, display the results and save the best models.
        """

        # To log fancy progress bar
        epoch_loop = tqdm(range(1 + self.start_epoch, self.parameters.epochs + 1))
        epoch_loop.set_description(f"Training for {self.parameters.epochs} epochs")  # Description

        start = time() - self.elapsed_time.seconds  #  For elapsed time
        for epoch_ndx in epoch_loop:

            # Train, update metrics, and return epoch train loss
            epoch_train_loss = self.train_epoch(epoch_ndx)
            
            if self.do_validation:
            
                # Validate, update validation metrics and return average mask and box IoU, for each class
                scores, iou_mask, iou_box = self.validate(epoch_ndx)
                iou_mean = np.append(iou_mask, iou_box).mean()  # Mean over four IoUs

                # Update progress bar
                epoch_loop.set_postfix_str(f"Train loss = {epoch_train_loss[-1]:.3f}, IoU (k, m) mask, box, mean = {np.round(iou_mask, 3)}, {np.round(iou_box, 3)}, {np.round(iou_mean, 3)}")
                
                # Update best model (according to custom conditions)
                mean_condition = iou_mean > self.best_iou_mean
                mask_membrane_condition = (iou_mean == self.best_iou_mean and iou_mask[1] > self.metrics_of_best['iou_mask_membrane'])
                if mean_condition or mask_membrane_condition:
                    # Metrics
                    self.metrics_of_best['score_kit'] = scores[0]
                    self.metrics_of_best['score_membrane'] = scores[1]
                    self.metrics_of_best['iou_mask_kit'] = iou_mask[0]
                    self.metrics_of_best['iou_mask_membrane'] = iou_mask[1]
                    self.metrics_of_best['iou_box_kit'] = iou_box[0]
                    self.metrics_of_best['iou_box_membrane'] = iou_box[1]

                    self.best_iou_mean = iou_mean
                    self.best_epoch = epoch_ndx
                    self.best_model = copy.deepcopy(self.model)

                    # Save model state
                    if save_model_bool:
                        self.save_model()
                    
                # Calculate elapsed time
                end = time()
                self.elapsed_time = timedelta(seconds=int(end-start))
                
                # Save full class state (in separate file if it is the best)
                if save_state_bool:
                    if mean_condition or mask_membrane_condition:
                        self.save_state(epoch_ndx, stamp='_best')
                    else:
                        self.save_state(epoch_ndx, stamp='')
                
            else:
                epoch_loop.set_postfix_str(f"Train loss = {epoch_train_loss[-1]:.3f}")

            # Update scheduler and log the change in learning rate
            if self.scheduler is not None:
                before_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                after_lr = self.optimizer.param_groups[0]["lr"]
                if after_lr != before_lr:
                    print(f"Epoch {epoch_ndx}: lr {before_lr} -> {after_lr}")

    def init_dataloaders(self):
        """
        Initialize train and validation dataloaders by splitting the original train dataset
        """

        print("Loading data...")
        
        loader_train = init_dataloader(
            self.data_args, self.kit_id,
            data_mode='train',
            n_batches=self.parameters.batch_size,
            n_workers=self.parameters.num_workers,
            shuffle=True,
            shots=self.shots,
            transform=self.transformation_train)

        # Validation set is obtained from 'val' or 'test' sets, spcecified in self.val_set
        if self.do_validation:
            loader_val = init_dataloader(
                self.data_args, self.kit_id,
                data_mode=self.val_set,
                n_batches=self.parameters.batch_size,
                n_workers=self.parameters.num_workers,
                shuffle=False,
                shots=[None]*len(self.shots),
                transform=self.transformation_val)
        else:
            loader_val = []

        return loader_train, loader_val


    def train_epoch(self, epoch_ndx):
        """
        Train model for one epoch and return training total loss
        """

        # To store running train metrics (loss)
        running_loss = np.zeros(len(self.metrics_train))

        # To log fancy progress bar
        train_loop = tqdm(self.loader_train, total=self.n_train)
        train_loop.set_description(f"Epoch {epoch_ndx} | Training")

        self.model.train()
        for images, targets in train_loop:

            # Send to device
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Set gradients to zero
            self.optimizer.zero_grad()

            # Forward pass and loss calculation (all in one)
            loss_dict = self.model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            # Backward pass
            total_loss.backward()

            # Optimize
            self.optimizer.step()

            # Update running loss
            running_loss += [v.item() for k, v in loss_dict.items()] + [total_loss.item()]

            # Update progress bar
            train_loop.set_postfix_str(f"Batch loss = {total_loss.item():.3f}")

        running_loss /= self.n_train
        
        # Update train metrics
        for j, k in enumerate(self.metrics_train.keys()):
            self.metrics_train[k].append(running_loss[j])
                
        return running_loss

    @torch.no_grad()
    def validate(self, epoch_ndx):
        """
        Validate the model over the entire validation set, calculate and update performance metrics (IoU)
        """

        # Set model to evaluation mode
        self.model.eval()

        # Final metrics over entire validation set
        scores_avg = np.zeros(self.n_classes)
        iou_masks_avg = np.zeros(self.n_classes)
        iou_boxes_avg = np.zeros(self.n_classes)

        # To log fancy progress bar
        val_loop = tqdm(self.loader_val, total=self.n_val)
        val_loop.set_description(f"Epoch {epoch_ndx} | Validation")

        for images, targets in val_loop:
            # Send to device and to list
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Inference step
            predictions = self.model(images)

            # Keep track of batch metrics to log
            batch_score = np.zeros(self.n_classes)
            batch_iou_mask = np.zeros(self.n_classes)
            batch_iou_box = np.zeros(self.n_classes)

            for img, target, pred in zip(images, targets, predictions):
                # Get labels and scores, which are aligned
                labels = pred['labels'].tolist()
                scores = pred['scores'].tolist()
                assert len(labels) == len(scores)

                # Get boxes and masks, which are also naturally aligned
                boxes = pred['boxes']
                masks = pred['masks']
                assert boxes.shape[0] == masks.shape[0] == len(labels)

                # Loop over classes excluding background (i.e. kit and membrane)
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
                        class_mask = (class_mask >= self.evaluation_args.mask_thresholds[i]).to(torch.uint8)

                        # Compute IoU
                        class_iou_mask = compute_iou_mask(class_mask, target['masks'][i])
                        class_iou_box = compute_iou_box(class_box, target['boxes'][i])

                        # Update dictionary
                        batch_score[i] += class_score
                        batch_iou_mask[i] += class_iou_mask
                        batch_iou_box[i] += class_iou_box

                    else:  # If there is no prediction, we leave all zeros in the dictionary
                        print(f'{cls} is missing from the prediction!')

            # Update overall metrics
            n_images = len(images)
            scores_avg += batch_score/n_images
            iou_masks_avg += batch_iou_mask/n_images
            iou_boxes_avg += batch_iou_box/n_images

            # Update progress bar
            val_loop.set_postfix_str(f"Scores = {np.round(batch_score/n_images, 3)} | IoU mask, box = {np.round(batch_iou_mask/n_images, 3)}, {np.round(batch_iou_box/n_images, 3)}")

        # Average metrics over loader length
        scores_avg /= self.n_val
        iou_masks_avg /= self.n_val
        iou_boxes_avg /= self.n_val

        # Update overall metrics
        self.metrics_val['score_kit'].append(scores_avg[0])
        self.metrics_val['score_membrane'].append(scores_avg[1])
        self.metrics_val['iou_mask_kit'].append(iou_masks_avg[0])
        self.metrics_val['iou_mask_membrane'].append(iou_masks_avg[1])
        self.metrics_val['iou_box_kit'].append(iou_boxes_avg[0])
        self.metrics_val['iou_box_membrane'].append(iou_boxes_avg[1])

        return scores_avg, iou_masks_avg, iou_boxes_avg


    def get_metrics(self):
        """
        Returns metrics as Pandas DataFrames ready to be plotted
        """

        metrics_val_df = pd.DataFrame.from_dict(self.metrics_val)
        metrics_val_df.index.name = 'epochs'

        metrics_train_df = pd.DataFrame.from_dict(self.metrics_train)
        metrics_train_df.index.name = 'epochs'

        return metrics_train_df, metrics_val_df

    @property
    def get_parameters(self):
        """
        Return parameters attribute but in dictionary format
        """
        return {k:v for k, v in self.parameters._get_kwargs()}

    def save_state(self, epoch, stamp):
        """
        Save whole instance state: model state, parameters, metrics, etc
        """
        state = {'args': self.args,
                 'parameters': {k:v for k, v in self.parameters._get_kwargs()},
                 'do_validation': self.do_validation,
                 'model_state': self.model.state_dict(),
                 'optimizer_state': self.optimizer.state_dict(),
                 'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,
                 'metrics_train': self.metrics_train,
                 'metrics_val': self.metrics_val, 
                 'file_path': self.file_path,
                 'epoch': epoch,
                 'elapsed_time': self.elapsed_time,
                 'best_iou_mean': self.best_iou_mean,
                 'metrics_of_best': self.metrics_of_best,
                 'best_model_state': self.best_model.state_dict()}
        torch.save(state, self.file_path + stamp + '.state')

    def save_model(self):
        """
        Save only model state 
        """
        torch.save({'model_state': self.model.state_dict()}, self.file_path + '_model.state')

    @classmethod
    def from_saved_state(cls, state_path, new_epochs=None):
        """
        Initialize Trainer class from previous saved state. This is useful, for instance, when training 
        was interrupted and we want to resume from where it stopped. 
        """

        print("Loading saved state...")
        
        state = torch.load(state_path, map_location=torch.device('cpu'))
        parameters = state['parameters']
        # Update new epochs in parameters dictionary
        if new_epochs is not None:
            if new_epochs <= parameters['epochs']:
              raise ValueError(f"new_epochs ({new_epochs}) must be bigger than current epochs ({parameters['epochs']})!")
            parameters['epochs'] = new_epochs
        
        print(f"Epoch: {state['epoch']}")
        print(f"Best IoU: {state['best_iou']}")
        print(f"Elapsed time: {state['elapsed_time']}")

        trainer = cls(args=state['args'], 
                      parameters=parameters, 
                      do_validation=state['do_validation'])
        
        # Load model, optimizer and scheduler states
        trainer.model.load_state_dict(state['model_state'])
        trainer.best_model.load_state_dict(state['best_model_state'])
        trainer.optimizer.load_state_dict(state['optimizer_state'])
        if trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(state['scheduler_state'])
        # Load metrics
        trainer.metrics_train = state['metrics_train']
        trainer.metrics_val = state['metrics_val']
        # To continue training from checkpoint correctly
        trainer.best_iou_mean = state['best_iou_mean']
        trainer.metrics_of_best = state['metrics_of_best']
        trainer.start_epoch = state['epoch']
        trainer.elapsed_time = state['elapsed_time']
        trainer.file_path = state['file_path']

        return trainer
