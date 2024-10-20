from PIL import Image
import os
import numpy as np
import logging
import yaml


# Configuration settings
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, config_path):
    with open(config_path, "w") as file:
        yaml.safe_dump(config, file)


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
def process_and_save_expt_artifacts(model_name, latents, image, output_dir, index):
    if model_name in ["ldm-celebahq-256"]:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype("uint8")
        image = Image.fromarray(image)

    else:
        logging.error("Save utils not setup for model: {model_name}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_image(output_dir, image, index)
    save_latents(output_dir, latents, index)
