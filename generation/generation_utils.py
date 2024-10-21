import torch
import logging


def run_denoising_loop(model_name, model_artifacts, latents):
    match model_name:
        case "ldm-celebahq-256":
            unet, vqvae, scheduler = model_artifacts

            with torch.no_grad():
                # Denoising loop
                for t in scheduler.timesteps:
                    # Predict noise residual
                    noise_pred = unet(latents, t).sample

                    # Update latents using scheduler
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

                # Decode the latents to get the image
                image = vqvae.decode(latents).sample
            return image, latents

        case _:
            logging.error(f"Unsupported denoising loop for model: {model_name}")
