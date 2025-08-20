import os
from omegaconf import OmegaConf
from sbgm.training_main import train_main


def run():
    # Register environment resolver to allow ${env:VAR_NAME} syntax in configuration files
    OmegaConf.register_new_resolver("env", lambda x: os.environ.get(x))
    
    # Path to default configuration file
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'full_run_config.yaml')
    config_path = os.path.abspath(config_path)

    # Load the configuration, ensuring it is in a container format and then creating a new OmegaConf object
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True) # Need to have resolve=True to ensure that all interpolations (i.e. ${...}) are resolved
    cfg = OmegaConf.create(cfg)

    print("Resolved data_dir:")
    print(cfg.paths.data_dir)
    # Launch the training process
    train_main(cfg)

