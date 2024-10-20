import logging


def load_model_files(model_cfg, device, inversion=False):
    model_name = model_cfg["name"]
    model_root = model_cfg["root"]

    match model_name:
        case "ldm-celebahq-256":
            logging.info(f"Loading VQ-VAE and UNet from: {model_name}")

            # Import model libraries
            from diffusers import UNet2DModel, VQModel
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            vqvae = VQModel.from_pretrained(
                model_root, subfolder=model_cfg["vqvae"]
            ).to(device)
            unet = UNet2DModel.from_pretrained(
                model_root, subfolder=model_cfg["unet"]
            ).to(device)
            scheduler_config = model_cfg["scheduler"]
            scheduler = DDIMScheduler(**scheduler_config)

            # Return inverse scheduler as well if invert is set to true
            if inversion:
                from diffusers.schedulers.scheduling_ddim import DDIMInverseScheduler

                inverse_scheduler = DDIMInverseScheduler.from_pretrained(
                    model_root, subfolder=model_cfg["scheduler"]
                )
                return vqvae, unet, scheduler, inverse_scheduler

            return vqvae, unet, scheduler

        case _:
            logging.error("Model setup information not available for: {model_name}")
