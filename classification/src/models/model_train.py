"""
File containing the TrainerClassification class, the main class for training and validating the classification model.
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from time import time
from datetime import datetime as dt
from datetime import timedelta
from tqdm.notebook import tqdm, trange  # Just to log fancy training progress bar

# Custom packages
from src.data.dataset import MembraneZonesDataset
from src.data.dataloader import init_dataloader
from src.models.model_architecture import ClassificationModel
from src.data.transformations import PreprocessMembraneZones, AugmentedMembraneZones
from src.utils.miscellaneous import set_seed


class TrainerClassification:
    def __init__(self, args, kit_id, do_validation=True, val_set='val', shots=None):

        # Assign attributes
        self.args = args
        self.kit_id = kit_id
        self.do_validation = do_validation
        self.val_set = val_set
        self.shots = shots

        # Split configuration file
        self.data_args = argparse.Namespace(**args.data_args)
        self.training_args = argparse.Namespace(**args.training_args)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.training_args['device'] = self.device
        print(f'Using {self.device} device')

        # Metrics
        self.metrics_train = {'loss': [], 'accuracy': []}
        self.metrics_val = {'loss': [], 'accuracy': []} if self.do_validation else None

        # Set seed for reproducibility
        set_seed(self.training_args.seed)

        # Transformations
        self.transformation_train = AugmentedMembraneZones()
        self.transformation_val = PreprocessMembraneZones()

        # Load training and validation datasets and dataloaders
        self.loader_train, self.loader_val = self.init_dataloaders()
        self.n_train = len(self.loader_train)
        self.n_val = len(self.loader_val)
        self.n_valset = len(self.loader_val.dataset)*self.loader_train.dataset.n_zones

        # Model architecture, criterion, optimizer, and scheduler
        self.model = ClassificationModel().to(self.device)
        self.optimizer = optim.Adam(
            params=self.model.parameters(), 
            lr=float(self.training_args.lr), 
            weight_decay=float(self.training_args.weight_decay))
        self.criterion = nn.BCELoss(reduction='mean')
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.training_args.scheduler_step, 
            gamma=self.training_args.scheduler_gamma)
        
        # Timestamp to identify training runs
        time_str = dt.now().strftime('%Y-%m-%d_%H.%M.%S')  
        self.file_path = f'{self.training_args.save_path}/{time_str}'
        
        # To continue training from checkpoint
        self.start_epoch = 0
        self.elapsed_time = timedelta(0)
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def train(self, save_bool=False):
        """
        Trains and validates the model for all epochs, store the training and
        validation metrics, log results and save the best models.
        """        

        # To log fancy progress bar
        epoch_loop = tqdm(range(1 + self.start_epoch, self.training_args.epochs + 1))
        epoch_loop.set_description(f"Training for {self.training_args.epochs} epochs")  # Description

        start = time() - self.elapsed_time.seconds  #  For elapsed time
        for epoch_ndx in epoch_loop:

            # Train, update train metric and return training loss
            epoch_train_loss, epoch_train_accuracy = self.train_epoch(epoch_ndx)
            
            if self.do_validation:
            
                # Validate, update validation metrics and return epoch loss and accuracy
                epoch_val_loss, epoch_val_accuracy = self.validate(epoch_ndx)
                
                if epoch_val_accuracy > self.best_accuracy:
                    self.best_accuracy = epoch_val_accuracy
                    self.best_epoch = epoch_ndx
                               
                # Number of missclassified examples
                n_wrong = self.n_valset - int(epoch_val_accuracy*self.n_valset)
                # Update progress bar
                epoch_loop.set_postfix_str(f"Train loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}, Train Acc = {(epoch_train_accuracy*100):.3f}, Val Acc = {(epoch_val_accuracy*100):.3f}, Ratio = {n_wrong}/{self.n_valset}")

                # Calculate elapsed time
                end = time()
                self.elapsed_time = timedelta(seconds=int(end-start))
            
                # Save model (in separate file if it is the best)
                if save_bool:
                    if epoch_val_accuracy == self.best_accuracy:
                        self.save_model(epoch_ndx, stamp='_best')
                    else:
                        self.save_model(epoch_ndx, stamp='')

            else:
                epoch_loop.set_postfix_str(f"Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}, Train Acc = {(epoch_train_accuracy*100):.3f}")             

            # Update scheduler and log the change in learning rate
            before_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            after_lr = self.optimizer.param_groups[0]["lr"]
            if after_lr != before_lr:
                print(f"Epoch {epoch_ndx}: lr {before_lr} -> {after_lr}")

    def init_dataloaders(self):
        """
        Initialize train and validation dataloaders.
        Batch_size is divided by the number of zones because of the way in which the dataset is retrieved
        and iterated over.
        """

        print("Loading data...")

        loader_train = init_dataloader(
            self.data_args, self.kit_id, 
            data_mode='train',
            n_batches=self.training_args.batch_size//2,
            n_workers=self.training_args.num_workers,
            shuffle=True,
            shots=self.shots, 
            transform=self.transformation_train)
               
        # Validation dataloader using 'val' or 'test' sets (as specified by self.val_set)
        if self.do_validation:
            loader_val = init_dataloader(
                self.data_args, self.kit_id, 
                data_mode=self.val_set,
                n_batches=self.training_args.batch_size//2,
                n_workers=self.training_args.num_workers,
                shuffle=False,
                shots=None, 
                transform=self.transformation_val)
        else:
            loader_val = []

        return loader_train, loader_val

    def train_epoch(self, epoch_ndx):
        """
        Train model for one epoch and update training metrics
        """

        # To store running train loss and accuracy
        running_loss = 0.0
        running_accuracy = 0.0

        # To log fancy progress bar
        train_loop = tqdm(self.loader_train, total=self.n_train)
        train_loop.set_description(f"Epoch {epoch_ndx} | Training")

        self.model.train()
        for x, y in train_loop:

            # Send to device
            x = x.to(self.device)
            y = y.to(self.device)
            # Set gradients to zero
            self.optimizer.zero_grad()
            # Forward pass
            y_pred = self.model(x)
            # Compute loss
            loss = self.criterion(y_pred, y)
            # Backward pass
            loss.backward()
            # Optimize
            self.optimizer.step()
            # Update running loss
            running_loss += loss.item()
            # Train accuracy
            with torch.no_grad():
                accuracy = self.get_accuracy(y_pred, y).item()
                running_accuracy += accuracy

            # Update progress bar
            train_loop.set_postfix_str(f"Loss = {loss.item():.4f} | Accuracy = {(accuracy*100):.3f}%")

        # Update train metric
        running_loss /= self.n_train
        running_accuracy /= self.n_train
        self.metrics_train['loss'].append(running_loss)
        self.metrics_train['accuracy'].append(running_accuracy)

        return running_loss, running_accuracy

    @torch.no_grad()
    def validate(self, epoch_ndx):
        """
        Validate the model over the entire validation set, calculate and update performance metrics
        """

        # Set model to evaluation mode
        self.model.eval()

        # Store epoch loss and accuracy
        running_loss = 0.0
        running_accuracy = 0.0

        # To log fancy progress bar
        val_loop = tqdm(self.loader_val, total=self.n_val)
        val_loop.set_description(f"Epoch {epoch_ndx} | Validation")

        for x, y in val_loop:
            # Send to device
            x = x.to(self.device)
            y = y.to(self.device)
            # Forward pass
            y_pred = self.model(x)
            # Compute loss
            loss = self.criterion(y_pred, y).item()
            # Compute accuracy
            accuracy = self.get_accuracy(y_pred, y).item()
            # Update running loss and accuracy
            running_loss += loss
            running_accuracy += accuracy
            # Update progress bar
            val_loop.set_postfix_str(f"Loss = {loss:.4f} | Accuracy = {(accuracy*100):.3f}%")
        
        # Update metrics
        running_loss /= self.n_val
        running_accuracy /= self.n_val
        self.metrics_val['loss'].append(running_loss)
        self.metrics_val['accuracy'].append(running_accuracy)

        return running_loss, running_accuracy

    @staticmethod
    def get_accuracy(y_pred, y):
        acc = np.mean((y.squeeze() == (y_pred.squeeze() > 0.5)).to(int).tolist())
        return acc

    def get_metrics(self):
        """
        Returns metrics as Pandas DataFrames ready to be plotted
        """

        metrics_val_df = pd.DataFrame.from_dict(self.metrics_val)
        metrics_val_df.index.name = 'epochs'

        metrics_train_df = pd.DataFrame.from_dict(self.metrics_train)
        metrics_train_df.index.name = 'epochs'

        return metrics_train_df, metrics_val_df

    def save_model(self, epoch_ndx, stamp):
        """
        Save model state, parameters, and metrics.
        """
        state = {'args': self.args,
                 'kit_id': self.kit_id,
                 'do_validation': self.do_validation,
                 'val_set': self.val_set,
                 'shots': self.shots,
                 'model_state': self.model.state_dict(),
                 'optimizer_state': self.optimizer.state_dict(),
                 'scheduler_state': self.scheduler.state_dict(),
                 'metrics_train': self.metrics_train,
                 'metrics_val': self.metrics_val, 
                 'file_path': self.file_path,
                 'epoch': epoch,
                 'elapsed_time': self.elapsed_time,
                 'best_accuracy': self.best_accuracy}
        torch.save(state, self.file_path + stamp + '.state')

    def save_txt_file(self):
        with open(self.file_path + '.txt', 'w') as file:
            file.write(str(self.args) + '\n\n')
            file.write(f'Best accuracy: {str(self.best_accuracy)}\n')
            file.write(f'Ratio: {str(int((1 - self.best_accuracy)*self.n_valset))}\n')
            file.write(f'Best epoch: {str(self.best_epoch)}\n')

    @classmethod
    def from_saved_state(cls, state_path, new_epochs=None):
        """
        Initialize Trainer class from previous saved state. This is useful, for instance, when training 
        was interrupted and we want to resume from where it stopped. 
        """

        print("Loading saved state...")
        
        state = torch.load(state_path, map_location=torch.device('cpu'))

        # Update new epochs in arg dictionary
        if new_epochs is not None:
            if new_epochs <= state['args'].training_args['epochs']:
              raise ValueError(f"new_epochs ({new_epochs}) must be bigger than current epochs ({state['args'].training_args['epochs']})!")
            state['args'].training_args['epochs'] = new_epochs
        
        print(f"Epoch: {state['epoch']}")
        print(f"Best accuracy: {state['best_accuracy']}")
        print(f"Elapsed time: {state['elapsed_time']}")

        trainer = cls(args=state['args'], 
                      kit_id=state['kit_id'], 
                      do_validation=state['do_validation'],
                      val_set=state['val_set'], 
                      shots=state['shots'])
        
        # Load model, optimizer and scheduler states
        trainer.model.load_state_dict(state['model_state'])
        trainer.optimizer.load_state_dict(state['optimizer_state'])
        trainer.scheduler.load_state_dict(state['scheduler_state'])
        # Load metrics
        trainer.metrics_train = state['metrics_train']
        trainer.metrics_val = state['metrics_val']
        # To continue training from checkpoint correctly
        trainer.best_accuracy = state['best_accuracy']
        trainer.start_epoch = state['epoch']
        trainer.elapsed_time = state['elapsed_time']
        trainer.file_path = state['file_path']

        return trainer

