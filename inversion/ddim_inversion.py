import os
import logging
from utils import process_and_save_expt_artifacts
import torch
from generation.generation_utils import run_denoising_loop
from tqdm import tqdm


class DDIMInversion:
    def __init__(self, config, model_artifacts, device, output_dir):
        self.config = config
        self.model_cfg = self.config["model"]
        self.model_artifacts = model_artifacts
        self.device = device
        self.output_dir = output_dir

    def load_model_artifacts(self, model_name):
        match model_name:
            case "ldm-celebahq-256":
                (
                    self.unet,
                    self.vqvae,
                    self.scheduler,
                    self.inverse_scheduler,
                ) = self.model_artifacts
                self.inverse_scheduler.set_timesteps(
                    self.model_cfg["inversion"]["num_inference_steps"]
                )
                self.scheduler.set_timesteps(
                    self.model_cfg["inversion"]["num_inference_steps"]
                )
            case _:
                logging.error(f"Unknown model artifacts for model :{model_name}")

    def unconditional_ddim_inversion(self, model_name, image_tensor):
        match model_name:
            case "ldm-celebahq-256":
                image_tensor = image_tensor

                # Encode to latents
                latents = self.vqvae.encode(image_tensor).latents

                # Inversion loop
                with torch.no_grad():
                    for t in reversed(self.inverse_scheduler.timesteps):
                        # Predict noise residual
                        noise_pred = self.unet(latents, t).sample

                        # Update latents using inverse scheduler
                        latents = self.inverse_scheduler.step(
                            noise_pred, t, latents
                        ).prev_sample

                # Use the learned noise to run DDIM generation to get reconstructed image
                reconstructed_image, reconstructed_image_latent = run_denoising_loop(
                    model_name, [self.unet, self.vqvae, self.scheduler], latents
                )

                return reconstructed_image, latents, reconstructed_image_latent

            case _:
                logging.error(f"Unsupported model: {model_name}")

    def run_ddim_inversion_loop(self, dataloader):
        model_cfg = self.config["model"]

        match model_cfg["type"]:
            case "unconditional":
                model_name = model_cfg["name"]
                steps = model_cfg["inversion"]["num_inference_steps"]
                self.load_model_artifacts(model_name)
                logging.info(
                    f"Starting Unconditional DDIM inversion with {steps} steps"
                )

                for image_tensor, image_path in tqdm(
                    dataloader, desc="Inverting images", unit="image"
                ):
                    image_tensor = image_tensor.to(self.device)
                    (
                        reconstructed_image,
                        learned_seed_latents,
                        reconstructed_image_latent,
                    ) = self.unconditional_ddim_inversion(model_name, image_tensor)

                    # Process and save the inverted image and latents
                    image_name = os.path.basename(image_path[0])
                    index = image_name.split("_")[-1].split(".")[
                        0
                    ]  # Extract the index from filename
                    process_and_save_expt_artifacts(
                        model_name,
                        [reconstructed_image, learned_seed_latents],
                        self.output_dir,
                        index,
                    )

            case _:
                logging.error(f"Unsupported inversion type : {model_cfg['type']}")