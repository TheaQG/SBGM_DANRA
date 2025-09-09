import os
from omegaconf import OmegaConf
from sbgm.evaluate_sbgm.generation_main import generation_main


def run(cfg):

    print("Resolved data_dir:")
    print(cfg.paths.data_dir)
    # Launch the training process
    generation_main(cfg)