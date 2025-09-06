import os 
import logging
import random
import numpy as np
import torch
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from sbgm.training_utils import get_model, get_gen_dataloader
from sbgm.special_transforms import build_back_transforms
from evaluate_sbgm.generation import SampleGenerator #run_generation_multiple, run_generation_single, run_generation_repeated
from sbgm.utils import get_model_string

def setup_logger(log_dir, name="gen_log", log_to_stdout=True):
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


def generation_main(cfg):
    """
    Main function to run generation from a trained model.
    """
    # Set seed
    torch.manual_seed(cfg.evaluation.seed)
    torch.cuda.manual_seed(cfg.evaluation.seed)
    np.random.seed(cfg.evaluation.seed)

    # Setup logging
    model_name_str = get_model_string(cfg)
    gen_dir = os.path.join(cfg["paths"]["sample_dir"], 'generation', model_name_str)
    log_gen_dir = os.path.join(gen_dir, 'logs')
    
    # Make sure dirs exist
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(log_gen_dir, exist_ok=True)

    logger = setup_logger(log_gen_dir)
    logger.info(f'[INFO] Configuration: {OmegaConf.to_yaml(cfg)}') # Print the configuration for debugging

    # --- 1. Set device -------------------------------------------------------------
    device = cfg.training.device

    # --- 2. Load model and data -------------------------------------------------------------
    model, ckpt_dir, ckpt_name = get_model(cfg)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    best_model_state = torch.load(ckpt_path, map_location=device)['network_params']
    model.load_state_dict(best_model_state)
    logger.info(f'[INFO] Model checkpoint loaded from: {ckpt_dir}/{ckpt_name}')

    # --- 3. Load generation dataloader ----------------------------------------------
    gen_dataloader = get_gen_dataloader(cfg)

    # --- 4. Prepare back transforms --------------------------------------------------------
    back_transforms = build_back_transforms(hr_var=cfg.highres.variable,
                                            hr_scaling_method= cfg.highres.scaling_method,
                                            hr_scaling_params=cfg.highres.scaling_params,
                                            lr_vars=cfg.lowres.condition_variables,
                                            lr_scaling_methods=cfg.lowres.scaling_methods,
                                            lr_scaling_params=cfg.lowres.scaling_params,
                                            )
    

    # --- Initialize SampleGenerator --------------------------------------------------------
    generator = SampleGenerator(cfg, model, gen_dataloader, back_transforms, device)

    # --- Choose generation type to run --------------------------------------------------------
    gen_types = cfg.evaluation.gen_type
    valid_types = {'multiple', 'single', 'repeated'}

    for gen_type in gen_types:
        if gen_type not in valid_types:
            raise ValueError(f"\nUnknown generation type: {gen_type}\n")
        
        logger.info(f"[INFO] Running generation type: {gen_type}")

        if gen_type == 'multiple':
            logger.info(f"[INFO] Running {cfg.evaluation.n_gen_samples} multiple generations...")
            generator.generate_multiple()
            logger.info("[INFO] Multiple generations completed.\n")
        elif gen_type == 'single':
            logger.info("[INFO] Running single generation...")
            generator.generate_single()
            logger.info("[INFO] Single generation completed.\n")
        elif gen_type == 'repeated':
            logger.info(f"[INFO] Running {cfg.evaluation.n_repeats} repeated generations...")
            generator.generate_repeated()
            logger.info("[INFO] Repeated generation completed.\n")











    # # --- 5. Choose generation method ---------------------------------------------
    # # Is a list of strings
    # gen_types = cfg.evaluation.gen_type

    # valid_types = {'multiple', 'single', 'repeated'}

    # for gen_type in gen_types:
    #     if gen_type not in valid_types:
    #         raise ValueError(f"\nUnknown generation type: {gen_type}\n")
        
    #     logger.info(f"Running generation: {gen_type}")

    #     if gen_type == 'multiple':
    #         logger.info("[INFO] Running multiple generations...")
    #         run_generation_multiple(cfg, gen_dataloader, model, back_transforms, device)
    #     elif gen_type == 'single':
    #         logger.info("[INFO] Running single generation...")
    #         run_generation_single(cfg, gen_dataloader, model, back_transforms, device)
    #     elif gen_type == 'repeated':
    #         logger.info("[INFO] Running repeated generation...")
    #         run_generation_repeated(cfg, gen_dataloader, model, back_transforms, device)
    


# if __name__ == "__main__":
#     cfg = hydra.compose(config_name="default_config")
#     main_generation(cfg)
