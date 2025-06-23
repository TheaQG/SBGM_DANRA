# sbgm/training_main.py
import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from sbgm.training_utils import get_units, get_cmaps, get_model_string, get_model, get_optimizer, get_dataloader, get_scheduler
from sbgm.utils import *
from sbgm.training import TrainingPipeline_general
from sbgm.score_unet import marginal_prob_std_fn, loss_fn

def train_main(cfg):
    """
    Main function to run the training process.
    
    Args:
        cfg (dict): Configuration dictionary containing all necessary parameters.
    """
    # Setup logging
    # setup_logging(cfg.logging)

    # Set units and colormaps
    hr_unit, lr_units = get_units(cfg)
    hr_cmap_name, lr_cmap_dict = get_cmaps(cfg)
    extra_cmap_dict = {"topo": "terrain", "lsm": "binary", "sdf": "coolwarm"}
  

    # Set path to figures, samples, losses
    save_str = get_model_string(cfg)
    path_samples = cfg['paths']['path_save'] + 'samples' + f'/Samples' + '__' + save_str
    path_losses = cfg['paths']['path_save'] + '/losses'
    path_figures = path_samples + '/Figures/'

    # Set device
    if cfg['training']['device'] == 'cuda':
        if torch.cuda.is_available():
            cfg['training']['device'] = torch.device('cuda')
            print(f"▸ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            cfg['training']['device'] = torch.device('cpu')
            print("▸ CUDA is not available, using CPU instead.")
    else:
        cfg['training']['device'] = torch.device('cpu')
        print("▸ Using CPU for training.")

    # Load data
    train_dataloader, val_dataloader = get_dataloader(cfg.data)


    # Examine sample from train dataloader (sample is full batch)
    print('\n')
    sample = train_dataloader.dataset[0]
    for key, value in sample.items():
        try:
            print(f'{key}: {value.shape}')
        except AttributeError:
            print(f'{key}: {value}')
    print('\n\n')

    fig, axs = plot_sample(sample,
                    hr_model = cfg['highres']['model'],
                    hr_units = hr_unit,
                    lr_model = cfg['lowres']['model'],
                    lr_units = lr_units,
                    var = cfg['highres']['variable'],
                    show_ocean = cfg['visualization']['show_ocean'],
                    hr_cmap = hr_cmap_name,
                    lr_cmap_dict = lr_cmap_dict,
                    extra_keys = ['topo', 'lsm', 'sdf'],
                    extra_cmap_dict = extra_cmap_dict
                    )
    # Save the figure
    SAVE_NAME = 'Initial_sample_plot.png'
    fig.savefig(os.path.join(path_samples, SAVE_NAME), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"▸ Saved initial sample plot to {path_samples}/{SAVE_NAME}")
    # Set PLOT_FIRST to False to avoid plotting every batch
    
    #Setup checkpoint path
    checkpoint_dir = os.path.join(cfg['paths']['path_save'], cfg['paths']['checkpoint_dir'])

    checkpoint_name = save_str + '.pth.tar'

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Define the seed for reproducibility, and set seed for torch, numpy and random
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Set torch to deterministic mode, meaning that the same input will always produce the same output
    torch.backends.cudnn.deterministic = False
    # Set torch to benchmark mode, meaning that the best algorithm will be chosen for the input
    torch.backends.cudnn.benchmark = True
    
    # Get the model
    model, checkpoint_path, checkpoint_name = get_model(cfg)
    model = model.to(cfg['training']['device'])

    # Get the optimizer
    optimizer = get_optimizer(cfg, model)

    # Get the learning rate scheduler (if applicable)
    scheduler = None
    if cfg['training']['scheduler'] is not None:
        scheduler = get_scheduler(cfg, optimizer)
        print(f"▸ Using learning rate scheduler: {cfg['training']['scheduler']}")
    else:
        print("▸ No learning rate scheduler specified, using default learning rate.")

    # Define the training pipeline
    pipeline = TrainingPipeline_general(model=model,
                                        loss_fn=loss_fn,
                                        marginal_prob_std_fn=marginal_prob_std_fn,
                                        optimizer=optimizer,
                                        device=cfg['training']['device'],
                                        lr_scheduler=scheduler,
                                        cfg=cfg
                                        )

    
    # Load checkpoint if it exists
    if cfg['training']['load_checkpoint'] and os.path.exists(checkpoint_path):
        print(f"▸ Loading pretrained weights from checkpoint {checkpoint_path}")

        pipeline.load_checkpoint(checkpoint_path, load_ema=cfg['training']['load_ema'],)
    else:
        print(f"▸ No checkpoint found at {checkpoint_path}. Starting training from scratch.")

    
    # If training on cuda, print device name and empty cache
    if cfg['training']['device'] == 'cuda' and torch.cuda.is_available():
        print(f"▸ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"▸ Model is using {torch.cuda.memory_allocated() / 1e9:.2f} GB of GPU memory.")
        print(f"▸ Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        print(f"▸ Number of parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"▸ Number of trainable parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad and p.requires_grad):,}")
        print(f"▸ Number of non-trainable parameters in model: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
        torch.cuda.empty_cache()
    else:
        print("▸ Using CPU for training.")
        print(f"▸ Number of parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"▸ Number of trainable parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad and p.requires_grad):,}")
        print(f"▸ Number of non-trainable parameters in model: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")








"""
    Write everything more modular:
        - get_dataloader (to get training, validation and possibly test dataloaders)
        - get_model (to get the model in the configuration wanted)
        - get_optimizer (to get the optimizer)
        - get_loss (to get the loss function)
        - get_scheduler (to get the learning rate scheduler)
        - train_model (to train the model)
        - evaluate_model (to evaluate the model)
        - save_model (to save the model)
        - load_model (to load the model)
        - plot_results (to plot the results)
        

"""



################################################################################################################################














