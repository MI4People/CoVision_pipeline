import torch
from torch.utils.data import DataLoader

from dataset_cla import MembraneZonesDataset

def init_dataloader(args, kit_id, data_mode, n_batches, n_workers, shuffle=False, shots=None, transform=None):
        """
        Initialize dataloader
        """

        dataset = MembraneZonesDataset(args,
                                       kit_id=kit_id,
                                       data_mode=data_mode,
                                       shots=shots,
                                       transform=transform)

        loader = DataLoader(dataset=dataset,
                            batch_size=n_batches,
                            shuffle=shuffle,
                            num_workers=n_workers,
                            collate_fn=collate_fn,
                            pin_memory=True)

        return loader

def collate_fn(batch):
    """
    Retrieve batches by concatenating batch and zone_num directions. Used in dataloaders.
    """
    x, y = zip(*batch)
    x = torch.cat(x)
    y = torch.cat(y).unsqueeze(1)
    
    return (x, y)