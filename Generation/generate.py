import torch
from utils import process_and_save_expt_artifacts
from Models.get_models import load_model_files
import logging


def unconditional_generation(model_cfg, num_images, device, output_dir):
    logging.info(f"Generating {num_images} images")
    model_name = model_cfg["name"]

    match model_name:
        case "ldm-celebahq-256":
            vqvae, unet, scheduler = load_model_files(model_cfg, device)
            for i in range(num_images):
                # Sample initial noise
                initial_seed = torch.randn(
                    (1, unet.config.in_channels, unet.sample_size, unet.sample_size)
                )
                latents = initial_seed.clone().to(device)
                scheduler.set_timesteps(model_cfg["generation"]["num_inference_steps"])

                with torch.no_grad():
                    # Denoising loop
                    for t in scheduler.timesteps:
                        # Predict noise residual
                        noise_pred = unet(latents, t).sample

                        # Update latents using scheduler
                        latents = scheduler.step(noise_pred, t, latents).prev_sample

                    # Decode the latents to get the image
                    image = vqvae.decode(latents).sample

                # Process and save the image and initial seed latents
                index = "0" * (len(str(num_images)) - len(str(i))) + str(i)
                process_and_save_expt_artifacts(
                    model_cfg["name"], initial_seed, image, output_dir, index
                )

        case _:
            logging.error(f"Unsupported model: {model_name}")


def run_generation(expt_cfg, output_dir, device, num_images):
    model_cfg = expt_cfg["model"]

    if model_cfg["type"] == "unconditional":
        unconditional_generation(model_cfg, num_images, device, output_dir)

    else:
        logging.error(f"Unsupported generation type : {model_cfg['type']}")
