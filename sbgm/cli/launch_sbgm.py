import os
from omegaconf import OmegaConf
from sbgm.training_main import train_main

def run():
    # Path to default configuration file
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
    config_path = os.path.abspath(config_path)

    # Load the configuration, ensuring it is in a container format and then creating a new OmegaConf object
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg)
    cfg = OmegaConf.create(cfg)

    # Launch the training process
    train_main(cfg)

