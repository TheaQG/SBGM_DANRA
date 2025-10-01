import logging
import yaml

from sbgm.evaluate_sbgm.evaluation_main import evaluation_main
from sbgm.evaluate_sbgm.plot_utils import make_publication_outputs

logger = logging.getLogger(__name__)

def run_evaluation(cfg, make_plots=True):


    # Launch the evaluation process
    evaluation_main(cfg)

    # Make publication-ready plots
    if make_plots:
        make_publication_outputs(cfg)
