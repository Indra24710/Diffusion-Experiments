import argparse
import torch
from Generation.generate import run_generation
from utils import load_config, save_config
from datetime import datetime
import logging
import os


# Set up logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image generation using diffusion models"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--num_images", type=int, default=1, help="Number of images to generate"
    )
    parser.add_argument(
        "--expt_name",
        type=str,
        required=True,
        help="Name of the experiment. Format:- modelname_experimentcount_someinfo. "
        "Example:- ldm-celebahq-256_expt-1_vanilla-image-generation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Parse args
    args = parse_args()

    # setup parameters
    expt_config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    date_str = datetime.now().strftime("%Y-%m-%d")
    experiment_name = date_str + "_" + args.expt_name
    output_dir = os.path.join("./experiments/generation/" + experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save config file for experiment reproducibility
    expt_config["other_experiment_info"] = {}
    expt_config["other_experiment_info"]["name"] = experiment_name
    expt_config["other_experiment_info"]["path"] = output_dir
    expt_config["other_experiment_info"]["num_images"] = args.num_images
    save_config(expt_config, os.path.join(output_dir, "expt_config.yml"))

    # Call generate function
    logging.info("Calling generate function")
    torch.manual_seed(expt_config["experiment"]["seed"])
    run_generation(expt_config, output_dir, device, args.num_images)
