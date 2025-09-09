from sbgm.evaluate_sbgm.evaluation_main import evaluation_main


def run(cfg):

    print("Resolved data_dir:")
    print(cfg.paths.data_dir)
    # Launch the evaluation process
    evaluation_main(cfg)
