import argparse
import torch
from inversion.ddim_inversion import DDIMInversion
from inversion.hybrid_inversion import HybridDDIMInversion
from utils import load_config, save_config, get_dataloader
from datetime import datetime
import logging
import os
from models.get_models import load_model_files


# Set up logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image inversion using diffusion models"
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
    output_dir = os.path.join("./experiments/inversion/" + experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tensorboard_dir = os.path.join(output_dir, "tensorboard_logs")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    # Save config file for experiment reproducibility
    cfg["other_experiment_info"] = {}
    cfg["other_experiment_info"]["name"] = experiment_name
    cfg["other_experiment_info"]["path"] = output_dir
    cfg["other_experiment_info"]["tensorboard_logs_path"] = tensorboard_dir
    save_config(cfg, os.path.join(output_dir, "expt_config.yml"))

    # Write the experiment info to a file
    output_file = os.path.join(output_dir, "experiment_info.txt")
    with open(output_file, "w") as f:
        f.write(args.experiment_info)

    # load image dataset
    dataloader = get_dataloader(cfg)

    expt_type = cfg["experiment"]["expt_type"]
    logging.info(f"Experiment type: {expt_type}")
    match expt_type:
        case "ddim_inversion":
            # Create ddim inversion object
            model_artifacts = load_model_files(cfg["model"], device, inversion=True)
            ddimInversionObj = DDIMInversion(cfg, model_artifacts, device, output_dir)

            # Call inv function
            logging.info("Calling inversion runner")
            torch.manual_seed(cfg["experiment"]["seed"])
            ddimInversionObj.run_ddim_inversion_loop(dataloader)

        case "hybrid_ddim_inversion":
            model_artifacts = load_model_files(cfg["model"], device, inversion=True)
            hybridDDIMInversionObj = HybridDDIMInversion(
                cfg, model_artifacts, device, output_dir
            )

            # Call inv function
            logging.info("Calling inversion runner")
            torch.manual_seed(cfg["experiment"]["seed"])
            hybridDDIMInversionObj.run_hybrid_ddim_inversion_loop(dataloader)

        case _:
            logging.error("Unknown inversion type")
