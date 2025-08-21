import os
import yaml
import torch
import logging
from datetime import datetime
import numpy as np
from omegaconf import OmegaConf

from sbgm.utils import get_model_string
from sbgm.evaluation import Evaluation

def setup_logger(log_dir, name="train_log", log_to_stdout=True):
    # Set up the path for the log directory
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Set up a logger, with level set to INFO which means it will log INFO, WARNING, ERROR, and CRITICAL messages
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers (we remove all handlers to avoid duplicates)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler to write logs to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    # Set the format for the log messages
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    # Apply the formatter to the file handler
    logger.addHandler(file_handler)

    # Optional: also print to terminal
    if log_to_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(file_formatter)
        logger.addHandler(stream_handler)

    logger.info(f"Logging to {log_path}")
    return logger


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
    log_gen_dir = os.path.join(gen_dir, 'logs')
    
    logger = setup_logger(log_gen_dir)
    logger.info(f'[INFO] Configuration: {OmegaConf.to_yaml(cfg)}') # Print the configuration for debugging



    # NEEDS TO BE MADE INTO A LOOP FOR POSSIBILITY OF ALL GENERATION TYPES
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
                    save_stats=cfg.get('save_stats', False),
                    n_samples=cfg.get('n_gen_samples', 4)
                )
            else:
                raise ValueError(f"Invalid evaluation method: {method}. Must be one of ['pixel_stats', 'spatial_stats']")
            logger.info(f'[INFO] Finished evaluation method: {method}\n')

        logger.info(f'[INFO] Finished evaluation for generated sample type: {gen_type}\n')

    logger.info('[INFO] Evaluation completed for all generated sample types!\n')



    # evaluation_multiple.spatial_statistics(show_figs=False, save_figs=True, save_stats=False, save_path=None, save_plot_path=args.path_save, n_samples=args.n_gen_samples)    
    



    # evaluation_single = Evaluation(args, generated_sample_type='single')


    # fig, axs = evaluation_single.plot_example_images(masked=False, plot_with_cond=True, plot_with_lsm=False, show_figs=False,save_figs=True, n_samples=4, same_cbar=False)    


    # evaluation_repeated = Evaluation(args, generated_sample_type='repeated')

    # fig, axs = evaluation_repeated.plot_example_images(masked=False, plot_with_cond=True, plot_with_lsm=False, show_figs=False, n_samples=4, save_figs=True)



# if __name__ == '__main__':
#     parser

