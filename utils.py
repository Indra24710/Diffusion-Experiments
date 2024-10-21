from PIL import Image
import os
import numpy as np
import logging
import yaml
from data.datasets.image_dataset import ImagesDataset
from torch.utils.data import DataLoader


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


def save_image(output_dir, image, index):
    if "images" not in os.listdir(output_dir):
        os.makedirs(os.path.join(output_dir, "images"))

    image_path = os.path.join(output_dir, "images", f"generated_image_{index}.png")
    image.save(image_path)
    return


def save_latents(output_dir, latents, index):
    latents = latents.detach().cpu().numpy()
    if "latents" not in os.listdir(output_dir):
        os.makedirs(os.path.join(output_dir, "latents"))
    save_path = os.path.join(output_dir, "latents", f"generated_latent_{index}.npz")
    np.savez(save_path, latents=latents)
    return


# Function to process and save images
def process_and_save_expt_artifacts(model_name, artifacts, output_dir, index):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    match model_name:
        case "ldm-celebahq-256":
            image, latents = artifacts
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype("uint8")
            image = Image.fromarray(image)
            save_image(output_dir, image, index)
            save_latents(output_dir, latents, index)

        case _:
            logging.error("Save utils not setup for model: {model_name}")
