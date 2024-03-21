import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

def show_dataset(args, kit_id):
    """
    Plot histogram for dataset of kit_id distinguishing train, test, and val sets.
    """
    
    # Read {kit_id}_labels.csv file
    labels_dir = os.path.join(args.data_dir, kit_id, f'{kit_id}_labels.csv')
    labels_df = pd.read_csv(labels_dir, index_col=0, dtype=str)
    
    # Count examples for each sequence configuration for each data mode
    counts = {'train': {}, 'test': {}, 'val': {}}
    for mode in ['train', 'test', 'val']:
        # Initialize all counts to zero
        counts[mode] = {k: 0 for k in args.sequences} 
        labels_count = labels_df[labels_df['data_mode']==mode]['line_sequence'].value_counts().items()
        for seq, count in labels_count:
            counts[mode][seq] = count
        print(mode, counts[mode])

    # Display histogram
    plt.figure(figsize=(15, 6))
    plt.bar(x=args.sequences, height=counts['train'].values(), label='train')
    plt.bar(x=args.sequences, height=counts['test'].values(), bottom=list(counts['train'].values()), label='test')
    plt.bar(x=args.sequences, height=counts['val'].values(), bottom=[c1 + c2 for c1, c2 in zip(counts['train'].values(), counts['test'].values())], label='val')
    plt.title(f'Number of membranes for each zone sequence for {kit_id} kit')
    plt.xlabel('Zone Sequence')
    plt.ylabel('N', rotation=0)
    plt.grid()
    plt.legend()
    plt.show()

def show_zones(zones, n_cols=2, mean=None, std=None):
    """
    Display cropped zones.
    """

    # Unormalize images
    if mean is not None and std is not None:
        mean_t = torch.Tensor(mean).reshape(1, -1, 1, 1)
        std_t = torch.Tensor(std).reshape(1, -1, 1, 1)
        zones = (zones * std_t) + mean_t    
    # Bring zones images to right format for cv2 (reverse channels from BGR (cv2) to RGB (plt))
    zones = (255*zones).to(torch.uint8).permute(0, 2, 3, 1).numpy()[:,:,:, ::-1]

    # Build the right grid for plotting multiple images
    n_zones = zones.shape[0]
    n_rows = n_zones // n_cols + 1 if n_zones % n_cols else n_zones // n_cols
    zones = [zones[i] if n_zones > i else None for i in range(n_rows * n_cols)]
  
    # Plot
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 2*n_rows))
    if n_zones != 1:
        ax = axs.flatten()[:len(zones)]
    else:
        ax = [axs]

    for i in range(n_zones):
        ax[i].imshow(zones[i])
        ax[i].axis('off')
    # Turn off the axis of the rest of subplots
    for j in range(n_zones, len(zones)):
        ax[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_metrics(metrics_train, metrics_val, figsize=(20, 10)):
    """
    Plot metrics as coming out from the training app.
    """

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize)

    ax0.plot(metrics_train['loss'], 'r-o', label='Train Loss')
    ax0.plot(metrics_val['loss'], 'b-o', label='Validation Loss')
    ax0.set_ylabel('Loss')
    ax0.set_xlabel('Epochs')
    ax0.legend()
    ax0.grid()

    ax1.plot(metrics_train['accuracy']*100, 'r-o', label='Train Accuracy (%)')
    ax1.plot(metrics_val['accuracy']*100, 'b-o', label='Validation Accuracy (%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    ax1.grid()
    
    plt.show()
