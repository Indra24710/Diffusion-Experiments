import os
import logging
from utils import (
    process_and_save_expt_artifacts,
    standardize_tensor,
    is_standard_normal_k2,
)
import torch
from tqdm import tqdm
from inversion.ddim_inversion import DDIMInversion
from generation.generation_utils import run_denoising_loop
import dnnlib
from inversion.inversion_utils import init_optimizer
from torch.utils.tensorboard import SummaryWriter
from loss_function_collection import LossFunctionCollection


class HybridDDIMInversion:
    def __init__(self, config, model_artifacts, device, output_dir):
        self.config = config
        self.model_cfg = self.config["model"]
        self.model_artifacts = model_artifacts
        self.device = device
        self.output_dir = output_dir
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.output_dir, "tensorboard_logs")
        )

    def load_model_artifacts(self, model_name):
        # Request model artifacts from the helper function based on model name
        match model_name:
            case "ldm-celebahq-256":
                self.ddimInversionObj.load_model_artifacts(model_name)
            case _:
                logging.error(f"Unknown model artifacts for model :{model_name}")

    # Unconditional hybrid inversion that combines DDIM inversion and optimization methods to improve inversion quality
    def unconditional_hybrid_ddim_inversion(self, model_name, target_image_tensor):
        match model_name:
            case "ldm-celebahq-256":
                # Setup for TensorBoard logging
                with torch.no_grad():
                    (
                        initial_ddim_inversion_image_li,
                        initial_ddim_inversion_latents_li,
                        _,
                    ) = self.ddimInversionObj.unconditional_ddim_inversion(
                        model_name, target_image_tensor
                    )

                initial_ddim_inversion_latents = initial_ddim_inversion_latents_li[0]
                initial_ddim_inversion_image = initial_ddim_inversion_image_li[0]
                logging.info(
                    f"Is initial latents from DDIM Inversion normal:- {is_standard_normal_k2(initial_ddim_inversion_latents.detach().cpu())[0]}"
                )

                # Get the initial noise tensor from vanilla DDIM inversion
                opt_latents = initial_ddim_inversion_latents.requires_grad_(True)
                logging.info(
                    f"Is initial latents from DDIM Inversion normal after standardization:- {is_standard_normal_k2(opt_latents.detach().cpu())[0]}"
                )

                target_image_features = self.vgg16(
                    target_image_tensor, resize_images=False, return_lpips=True
                )
                optimizer = init_optimizer(self.model_cfg, opt_latents)

                reconstructed_image = None
                reconstructed_image_latents = None
                loss_map = {
                    "main_loss": [],
                    "normality_loss": [],
                    "total_loss": [],
                    "normality_delta": [],
                }
                normal_latents = []

                # Run optimization process to improve the quality of the noise vector
                for epoch in tqdm(
                    range(self.model_cfg["inversion"]["optimization_steps"])
                ):
                    with torch.no_grad():
                        # opt_latents.copy_(normalize_tensor(opt_latents))
                        is_normal, is_normal_delta = is_standard_normal_k2(
                            opt_latents.detach().cpu()
                        )
                        loss_map["normality_delta"].append(is_normal_delta)
                        self.writer.add_scalar(
                            "Normality Delta", is_normal_delta, epoch
                        )
                        logging.info(
                            f"Is latents normalized:- {is_normal}, delta:- {is_normal_delta}"
                        )

                    # Get initial
                    (
                        reconstructed_image,
                        reconstructed_image_latents,
                    ) = run_denoising_loop(
                        model_name,
                        [
                            self.ddimInversionObj.unet,
                            self.ddimInversionObj.vqvae,
                            self.ddimInversionObj.scheduler,
                        ],
                        opt_latents,
                    )
                    reconstructed_image_features = self.vgg16(
                        reconstructed_image, resize_images=False, return_lpips=True
                    )

                    # Compute loss
                    total_loss, loss_details = self.lossesObj.compute_combined_loss(
                        opt_latents,
                        [target_image_tensor, reconstructed_image],
                        [target_image_features, reconstructed_image_features],
                    )

                    logging.info(f"epoch :- {epoch} loss :- {total_loss}")

                    is_normal, is_normal_delta = is_standard_normal_k2(
                        opt_latents.detach().cpu()
                    )

                    if is_normal:
                        normal_latents.append(
                            [epoch, total_loss, opt_latents.detach().cpu().numpy()]
                        )

                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    optimizer.step()

                # generate image from learned latents
                with torch.no_grad():
                    # Generate image using the latents which are normal and which also has the smallest total loss value
                    if len(normal_latents) > 0:
                        normal_latents.sort(key=lambda x: (x[1], -x[0]))
                        (
                            final_reconstructed_image,
                            final_reconstructed_image_latents,
                        ) = run_denoising_loop(
                            model_name,
                            [
                                self.ddimInversionObj.unet,
                                self.ddimInversionObj.vqvae,
                                self.ddimInversionObj.scheduler,
                            ],
                            torch.from_numpy(normal_latents[0][2]).to(self.device),
                        )
                        self.writer.close()
                        return (
                            final_reconstructed_image,
                            initial_ddim_inversion_image,
                            normal_latents,
                            final_reconstructed_image_latents,
                            initial_ddim_inversion_latents,
                            loss_map,
                        )
                    else:
                        (
                            final_reconstructed_image,
                            final_reconstructed_image_latents,
                        ) = run_denoising_loop(
                            model_name,
                            [
                                self.ddimInversionObj.unet,
                                self.ddimInversionObj.vqvae,
                                self.ddimInversionObj.scheduler,
                            ],
                            opt_latents.detach(),
                        )
                        self.writer.close()
                        return (
                            final_reconstructed_image,
                            initial_ddim_inversion_image,
                            opt_latents.detach().cpu().numpy(),
                            final_reconstructed_image_latents,
                            initial_ddim_inversion_latents,
                            loss_map,
                        )

            case _:
                logging.error(f"Unsupported model: {model_name}")

    def run_hybrid_ddim_inversion_loop(self, dataloader):
        model_cfg = self.config["model"]
        self.ddimInversionObj = DDIMInversion(
            self.config, self.model_artifacts, self.device, self.output_dir
        )
        self.lossesObj = LossFunctionCollection()
        self.lossesObj.load_from_config(model_cfg["inversion"]["losses"])

        # load vgg16 for perceptual loss
        url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f).eval().to(self.device)

        match model_cfg["type"]:
            case "unconditional":
                model_name = model_cfg["name"]
                steps = model_cfg["inversion"]["num_inference_steps"]
                self.load_model_artifacts(model_name)
                logging.info(
                    f"Starting Unconditional Hybrid DDIM inversion with {steps} steps"
                )

                # Call unconditional hybrid ddim inversion on each image you want to invert
                for image_tensor, image_path in tqdm(
                    dataloader, desc="Inverting images", unit="image"
                ):
                    image_tensor = image_tensor.to(self.device)

                    (
                        reconstructed_image,
                        initial_ddim_inversion_image,
                        normal_latents,
                        reconstructed_image_latent,
                        initial_ddim_inversion_latents,
                        loss_map,
                    ) = self.unconditional_hybrid_ddim_inversion(
                        model_name, image_tensor
                    )

                    # Process and save the inverted image and latents
                    image_name = os.path.basename(image_path[0])
                    index = image_name.split("_")[-1].split(".")[
                        0
                    ]  # Extract the index from filename
                    process_and_save_expt_artifacts(
                        model_name,
                        [
                            [reconstructed_image, initial_ddim_inversion_image],
                            [
                                normal_latents,
                                initial_ddim_inversion_latents.detach().cpu().numpy(),
                            ],
                            loss_map,
                        ],
                        self.output_dir,
                        index,
                    )

            case _:
                logging.error(f"Unsupported inversion type : {model_cfg['type']}")
