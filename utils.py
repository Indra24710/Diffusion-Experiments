from PIL import Image
import os
import numpy as np
import logging
import yaml
from data.datasets.image_dataset import ImagesDataset
from torch.utils.data import DataLoader
import torch
from scipy import stats
import json
from scipy.stats import kstest, normaltest
import pickle


# Configuration settings
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


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


def save_latents(output_dir, latents, filename):
    if "latents" not in os.listdir(output_dir):
        os.makedirs(os.path.join(output_dir, "latents"))
    save_path = os.path.join(output_dir, "latents", filename)
    with open(save_path, "wb") as f:
        pickle.dump(latents, f)
    return


# Function to process and save images
def process_and_save_expt_artifacts(model_name, artifacts, output_dir, index):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if "images" not in os.listdir(output_dir):
        os.makedirs(os.path.join(output_dir, "images"))

    # De-normalize the tensor
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])

    # Unsqueeze the mean and std to match the tensor shape [C,1,1] for broadcasting
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)

    match model_name:
        case "ldm-celebahq-256":
            images, latents, loss_map = artifacts

            if images:
                for image in images[1:]:
                    # process and save images
                    image = image.cpu() * std + mean
                    image = (image).clamp(0, 1)
                    image = image.permute(0, 2, 3, 1).numpy()[0]
                    image = (image * 255).astype("uint8")
                    image = Image.fromarray(image)
                    image_path = os.path.join(
                        output_dir, "images", f"generated_image_{index}.png"
                    )
                    image.save(image_path)

            if latents:
                # save latents
                if len(latents[0]) == 3:
                    save_latents(
                        output_dir, latents[0], f"generated_normal_latents_{index}.pkl"
                    )
                else:
                    save_latents(
                        output_dir, latents[0], f"generated_latents_{index}.pkl"
                    )

                if len(latents) > 1:
                    save_latents(
                        output_dir, latents[1], f"generated_ddim_latents_{index}.pkl"
                    )

            if loss_map:
                # save losses
                losses_file = open(os.path.join(output_dir, "losses.json"), "w")
                json.dump(loss_map, losses_file)
                losses_file.close()

        case _:
            logging.error("Save utils not setup for model: {model_name}")


def is_standard_normal_k2(tensor, alpha: float = 0.05):
    standardized_tensor = None

    if torch.is_tensor(tensor):
        # Ensure tensor is a 1D array for testing
        if len(tensor.shape) != 1:
            tensor = tensor.flatten()

        # Standardize the tensor (zero mean, unit variance)
        standardized_tensor = (
            (tensor - torch.mean(tensor)) / torch.std(tensor)
        ).numpy()
    else:
        standardized_tensor = tensor.flatten()

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
