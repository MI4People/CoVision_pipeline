from torch.utils.data import DataLoader

from dataset_seg import LFASegmentationDataset

def collate_fn(batch):
    """
    Sets format in which batches are retrieved
    """
    return tuple(zip(*batch))

def init_dataloader(args, kit_id, data_mode, n_batches, n_workers, shuffle=False, shots=None, transform=None):
        """
        Initialize dataloader
        """

        dataset = LFASegmentationDataset(args,
                                         kit_id=kit_id,
                                         data_mode=data_mode,
                                         shots=shots,
                                         transforms=transform)

        loader = DataLoader(dataset=dataset,
                            batch_size=n_batches,
                            shuffle=shuffle,
                            num_workers=n_workers,
                            collate_fn=collate_fn,
                            pin_memory=True)

        return loader