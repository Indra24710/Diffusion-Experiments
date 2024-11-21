import argparse
import torch
from generation.generate import GenerateImages
from utils import load_config, save_config
from models.get_models import load_model_files
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
        "--expt_name",
        type=str,
        required=True,
        help="Name of the experiment. Format:- modelname_experimentcount_someinfo. "
        "Example:- ldm-celebahq-256_expt-1_vanilla-image-generation",
    )
    parser.add_argument(
        "--experiment_info",
        type=str,
        required=True,
        help="A brief description or notes about the experiment.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Parse args
    args = parse_args()

    # setup parameters
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    date_str = datetime.now().strftime("%Y-%m-%d")
    experiment_name = date_str + "_" + args.expt_name
    output_dir = os.path.join("./experiments/generation/" + experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save config file for experiment reproducibility
    cfg["other_experiment_info"] = {}
    cfg["other_experiment_info"]["name"] = experiment_name
    cfg["other_experiment_info"]["path"] = output_dir
    save_config(cfg, os.path.join(output_dir, "expt_config.yml"))

    # Write the experiment info to a file
    output_file = os.path.join(output_dir, "experiment_info.txt")
    with open(output_file, "w") as f:
        f.write(args.experiment_info)

    # Call generate function
    logging.info("Calling generate function")
    torch.manual_seed(cfg["experiment"]["seed"])
    model_artifacts = load_model_files(cfg["model"], device)
    genImagesObject = GenerateImages(cfg, model_artifacts, device, output_dir)
    genImagesObject.run_generation()
