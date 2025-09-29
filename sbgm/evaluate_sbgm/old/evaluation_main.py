"""
    TODO:
        - Make sure that evaluation figures saved in SBGM_SD/models_and_samples/
"""


import os
import torch
import logging
from datetime import datetime
import numpy as np
from omegaconf import OmegaConf

from sbgm.utils import get_model_string
from sbgm.evaluate_sbgm.old.evaluation import Evaluation

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


def evaluation_main(cfg):
    """
    Main function to run evaluation from generated samples.
    """
    # Set seed
    torch.manual_seed(cfg.evaluation.seed)
    torch.cuda.manual_seed(cfg.evaluation.seed)
    np.random.seed(cfg.evaluation.seed)

    # Setup logging
    model_name_str = get_model_string(cfg)
    gen_dir = os.path.join(cfg["paths"]["sample_dir"], 'generation', model_name_str)

    # logger.info(f'[INFO] Configuration: {OmegaConf.to_yaml(cfg)}') # Print the configuration for debugging


    for gen_type in cfg.evaluation.get('eval_gen_type', ['multiple']):
        logger.info(f'[INFO] Running evaluation for generated sample type: {gen_type}\n')

        if gen_type not in ['single', 'repeated', 'multiple']:
            raise ValueError(f"Invalid generated sample type: {gen_type}. Must be one of ['single', 'repeated', 'multiple']")
        
        if gen_type == 'multiple':
            n_samples = cfg.evaluation.batch_size
        elif gen_type == 'single':
            n_samples = 1
        elif gen_type == 'repeated':
            n_samples = cfg.evaluation.n_repeats
        else:
            raise ValueError(f"Invalid generated sample type: {gen_type}. Must be one of ['single', 'repeated', 'multiple']")
        
        eval_runner = Evaluation(cfg=cfg, generated_sample_type=gen_type, n_samples=n_samples)

        if cfg.evaluation.get('plot_examples', True):
            logger.info(f'[INFO] Plotting example images for generated sample type: {gen_type}\n')
            fig, axs = eval_runner.plot_example_images(
                            masked=cfg.evaluation.mask_plots,
                            plot_with_cond=cfg.evaluation.plot_w_cond,
                            plot_with_lsm=cfg.evaluation.plot_w_lsm,
                            show_figs=cfg.evaluation.show_plots,
                            n_samples=cfg.evaluation.batch_size,
                            same_cbar=False,
                            save_figs=True
                            )

        for method in cfg.evaluation.get('eval_stat_methods', ['pixel_stats', 'spatial_stats']):
            logger.info(f'[INFO] Running evaluation method: {method}\n')
            if method == 'pixel_stats':
                eval_runner.full_pixel_statistics(
                    show_figs=cfg.evaluation.get('show_plots', False),
                    save_figs=cfg.evaluation.get('save_figs', True),
                    save_stats=cfg.evaluation.get('save_stats', False),
                    n_samples=cfg.evaluation.get('batch_size', 4)
                )
            elif method == 'spatial_stats':
                eval_runner.spatial_statistics(
                    show_figs=cfg.get('show_figs', False),
                    save_figs=cfg.get('save_figs', True),
                    n_samples=cfg.get('n_gen_samples', 4)
                )
            else:
                raise ValueError(f"Invalid evaluation method: {method}. Must be one of ['pixel_stats', 'spatial_stats']")
            logger.info(f'[INFO] Finished evaluation method: {method}\n')

        logger.info(f'[INFO] Finished evaluation for generated sample type: {gen_type}\n')

    logger.info('[INFO] Evaluation completed for all generated sample types!\n')

