import os 
import hydra
from omegaconf import DictConfig, OmegaConf

from sbgm.training_utils import get_model, get_dataloader
from special_transforms import build_back_transforms
from sbgm.generation import run_generation_multiple, run_generation_single, run_generation_repeated

@hydra.main(version_base=None, config_path="config", config_name="default_config")
def main(cfg: DictConfig):
    """
    Main function to run generation from a trained model.
    """

    print(OmegaConf.to_yaml(cfg)) # Print the configuration for debugging

    # --- 1. Set seed and device -------------------------------------------------------------
    seed = cfg.evaluation.seed
    device = cfg.training.device
    hydra.utils.set_seed(seed)

    # --- 2. Prepare directories and paths -----------------------------------
    path_save = cfg.paths.path_save
    path_samples = os.path.join(path_save, 'samples', cfg.experiment.name)
    path_figures = os.path.join(path_samples, 'Figures')
    os.makedirs(path_figures, exist_ok=True)

    # --- 3. Load model and data -------------------------------------------------------------
    model = get_model(cfg)
    _, _, gen_dataloader = get_dataloader(cfg)

    # --- 4. Prepare back transforms --------------------------------------------------------
    back_transforms = build_back_transforms(hr_var=cfg.highres.variable,
                                            hr_scaling_method= cfg.highres.scaling_method,
                                            hr_scaling_params=cfg.highres.scaling_params,
                                            lr_vars=cfg.lowres.condition_variables,
                                            lr_scaling_methods=cfg.lowres.scaling_methods,
                                            lr_scaling_params=cfg.lowres.scaling_params,
                                            )
    
    # --- 5. Choose generation method ---------------------------------------------
    gen_type = cfg.generation.gen:type

    if gen_type == 'multiple':
        print("Running multiple generations...")
        run_generation_multiple(cfg, gen_dataloader, model, back_transforms, device)
    elif gen_type == 'single':
        print("Running single generation...")
        run_generation_single(cfg, gen_dataloader, model, back_transforms, device)
    elif gen_type == 'repeated':
        print("Running repeated generation...")
        run_generation_repeated(cfg, gen_dataloader, model, back_transforms, device)
    else:
        raise ValueError(f"Unknown generation type: {gen_type}")
    

if __name__ == "__main__":
    cfg = hydra.compose(config_name="default_config")
    main(cfg)
