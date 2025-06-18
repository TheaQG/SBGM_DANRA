# sbgm/training_main.py

from sbgm.training import train_model
from sbgm.data_modules import get_dataloader
from sbgm.utils import setup_logging, load_config
from sbgm.models import get_model

def train_main(cfg):
    """
    Main function to run the training process.
    
    Args:
        cfg (dict): Configuration dictionary containing all necessary parameters.
    """
    # Setup logging
    setup_logging(cfg.logging)

    # Load data
    train_loader, val_loader = get_dataloader(cfg.data)

    # Train the model
    train_model(cfg, train_loader, val_loader)






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














