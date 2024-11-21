from PIL import Image
import os
import logging
import yaml
from data.datasets.image_dataset import ImagesDataset
from torch.utils.data import DataLoader
import torch
import json
from scipy.stats import normaltest
import pickle
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants for de-normalizing images
MEAN = torch.tensor([0.5, 0.5, 0.5])
STD = torch.tensor([0.5, 0.5, 0.5])
MEAN = MEAN.view(1, 3, 1, 1)
STD = STD.view(1, 3, 1, 1)


def save_config(config, config_path):
    with open(config_path, "w") as file:
        yaml.safe_dump(config, file)


def get_dataloader(config):
    expt_cfg = config["experiment"]
    dataset_cfg = config["dataset"]
    path = dataset_cfg["path"]

    match dataset_cfg["dataset_type"]:
        case "vanilla":
            img_dataset = ImagesDataset(path, dataset_cfg, expt_cfg["num_files"])
            img_dataloader = DataLoader(
                img_dataset,
                batch_size=expt_cfg["batch_size"],
                shuffle=expt_cfg["shuffle_dataset"],
            )
            return img_dataloader

        case _:
            logging.error("Dataset type not configured")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise


def save_latents(output_dir: str, latents: Any, filename: str) -> None:
    """
    Save latents to a file in the specified directory.

    Args:
        output_dir (str): The base directory for saving outputs.
        latents (Any): The latent representations to save.
        filename (str): The filename for the saved latents.
    """
    latents_dir = os.path.join(output_dir, "latents")
    os.makedirs(latents_dir, exist_ok=True)
    save_path = os.path.join(latents_dir, filename)
    try:
        with open(save_path, "wb") as f:
            pickle.dump(latents, f)
        logging.info(f"Latents saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save latents: {e}")


def process_and_save_image(
    image_tensor: torch.Tensor, output_dir: str, filename: str
) -> None:
    """
    De-normalize, process, and save an image tensor.

    Args:
        image_tensor (torch.Tensor): The image tensor to process and save.
        output_dir (str): The directory to save the image in.
        filename (str): The filename for the saved image.
    """
    # try:
    print(image_tensor.shape)
    image = image_tensor.cpu() * STD + MEAN
    image = image.clamp(0, 1)
    image = image.permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype("uint8")
    image = Image.fromarray(image)
    image_path = os.path.join(output_dir, filename)
    image.save(image_path)
    logging.info(f"Image saved to {image_path}")
    # except Exception as e:
    #     logging.error(f"Failed to process and save image: {e}")


def save_loss_map(loss_map: Dict, output_dir: str) -> None:
    """
    Save the loss map to a JSON file in the specified directory.

    Args:
        loss_map (Dict): The loss map data to save.
        output_dir (str): The directory to save the loss map in.
    """
    losses_path = os.path.join(output_dir, "losses.json")
    try:
        with open(losses_path, "w") as f:
            json.dump(loss_map, f)
        logging.info(f"Loss map saved to {losses_path}")
    except Exception as e:
        logging.error(f"Failed to save loss map: {e}")


def construct_artifacts(
    images: Optional[Dict[str, Any]] = None,
    latents: Optional[Dict[str, Any]] = None,
    loss_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Construct an artifacts dictionary from the provided parameters.

    Args:
        images (Optional[List[torch.Tensor]]): A list of image tensors.
        latents (Optional[Dict[str, Any]]): A dictionary of latent representations.
        loss_map (Optional[Dict[str, Any]]): A dictionary containing loss map data.

    Returns:
        Dict[str, Any]: A dictionary containing all the provided artifacts.
    """
    artifacts = {}

    if images is not None:
        if isinstance(images, dict) and images:
            artifacts["images"] = images
            logging.info("Images added to artifacts.")
        else:
            logging.warning("Images parameter is empty or not a list.")

    if latents is not None:
        if isinstance(latents, dict) and latents:
            artifacts["latents"] = latents
            logging.info("Latents added to artifacts.")
        else:
            logging.warning("Latents parameter is empty or not a dictionary.")

    if loss_map is not None:
        if isinstance(loss_map, dict) and loss_map:
            artifacts["loss_map"] = loss_map
            logging.info("Loss map added to artifacts.")
        else:
            logging.warning("Loss map parameter is empty or not a dictionary.")

    if not artifacts:
        logging.warning("No artifacts were provided to construct.")

    return artifacts


def process_and_save_expt_artifacts(
    artifacts: Dict[str, Any], output_dir: str, index: int, config: Dict[str, Any]
) -> None:
    """
    Process and save experiment artifacts such as images, latents, and loss maps.

    Args:
        artifacts (Dict[str, Any]): A dictionary containing various artifacts.
        output_dir (str): The base directory for saving outputs.
        index (int): An index to uniquely identify saved files.
        config (Dict[str, Any]): The experiment configuration dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Determine which artifacts to process based on the config
    expt_type = config.get("experiment", {}).get("expt_type", "default")
    model_name = config.get("model", {}).get("name", "unknown_model")

    # Process images if present
    if "images" in artifacts:
        images = artifacts["images"]
        if isinstance(images, dict):
            for key, image in images.items():
                image_filename = f"generated_image_{index}_{key}.png"
                process_and_save_image(image, images_dir, image_filename)

    # Process latents if present
    if "latents" in artifacts:
        latents = artifacts["latents"]
        if isinstance(latents, dict):
            for key, latent in latents.items():
                latents_filename = f"{key}_{index}.pkl"
                save_latents(output_dir, latent, latents_filename)
        else:
            latents_filename = f"latents_{index}.pkl"
            save_latents(output_dir, latents, latents_filename)

    # Process loss map if present
    if "loss_map" in artifacts:
        loss_map = artifacts["loss_map"]
        save_loss_map(loss_map, output_dir)

    logging.info(
        f"Artifacts processed for model: {model_name}, experiment type: {expt_type}"
    )


def is_standard_normal_k2(tensor, alpha: float = 0.05):
    standardized_tensor = None

    # Ensure tensor is a 1D array for testing
    if len(tensor.shape) != 1:
        tensor = tensor.flatten()

    # Standardize the tensor (zero mean, unit variance)
    standardized_tensor = ((tensor - torch.mean(tensor)) / torch.std(tensor)).numpy()

    # Perform the K^2 test against the standard normal distribution
    ks_stat, p_value = normaltest(standardized_tensor, nan_policy="raise")

    # Return True if p-value > alpha (fail to reject null hypothesis)
    return p_value > alpha, p_value - alpha


def standardize_tensor(tensor):
    # Compute the mean and standard deviation
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    # Normalize the tensor
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor
