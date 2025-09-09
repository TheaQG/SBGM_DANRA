import os
from omegaconf import OmegaConf
from sbgm.training_main import train_main


def run(cfg):

    print("Resolved data_dir:")
    print(cfg.paths.data_dir)
    # Launch the training process
    train_main(cfg)

