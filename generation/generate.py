import torch
from utils import process_and_save_expt_artifacts
import logging
from tqdm import tqdm
from generation.generation_utils import run_denoising_loop


class GenerateImages:
    def __init__(self, config, model_artifacts, device, output_dir):
        self.config = config
        self.model_cfg = self.config["model"]
        self.model_artifacts = model_artifacts
        self.device = device
        self.output_dir = output_dir
        self.load_model_artifacts(self.model_cfg["name"])

    def load_model_artifacts(self, model_name):
        # Request model artifacts from the helper function based on model name
        match model_name:
            case "ldm-celebahq-256":
                self.unet, self.vqvae, self.scheduler = self.model_artifacts
                self.scheduler.set_timesteps(
                    self.model_cfg["generation"]["num_inference_steps"]
                )

            case _:
                logging.error(f"Unsupported model: {model_name}")

    def unconditional_generation(self, model_name):
        match model_name:
            # Call the right diffusion model generation algorithm based on model name
            case "ldm-celebahq-256":
                # Sample initial noise vector which will act as the seed for the image generation process
                initial_seed_latent = torch.randn((1, 4, 32, 32))
                print("seed latent shape:- ", initial_seed_latent.shape)
                latents = initial_seed_latent.clone().to(self.device)

                # Run denoising loop on the sampled noise vector
                image, _ = run_denoising_loop(
                    model_name, [self.unet, self.vqvae, self.scheduler], latents
                )
                return image, initial_seed_latent

            case _:
                logging.error(f"Unsupported model: {model_name}")

    def run_generation(self):
        # Call the right generation process (unconditional/class-conditioned/text-conditioned) based on
        # model config type
        match self.model_cfg["type"]:
            case "unconditional":
                model_name = self.model_cfg["name"]
                num_images = self.config["experiment"]["num_files"]
                logging.info(f"Generating {num_images} images")
                for i in tqdm(range(num_images)):
                    with torch.no_grad():
                        # Call unconditional image generation process
                        image, initial_seed_latent = self.unconditional_generation(
                            model_name
                        )

                        # Process and save the image and initial seed latents
                        index = "0" * (len(str(num_images)) - len(str(i))) + str(i)
                        process_and_save_expt_artifacts(
                            model_name,
                            [image, initial_seed_latent],
                            self.output_dir,
                            index,
                        )

            case _:
                logging.error(f"Unsupported generation type : {self.model_cfg['type']}")
