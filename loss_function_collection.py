import yaml
import torch
import torch.nn.functional as F
import logging


# Custom loss function class to test different combinations of loss functions for custom experiments
class LossFunctionCollection:
    def __init__(self):
        self.loss_functions = {}
        self.loss_weights = {}

    def register_loss(self, name, loss_fn, input_type, weight=1.0):
        if name in self.loss_functions:
            raise ValueError(f"Loss function '{name}' is already registered.")
        self.loss_functions[name] = [input_type, loss_fn]
        self.loss_weights[name] = weight

    def compute_combined_loss(self, latents, images, image_features):
        target_image, recon_image = images
        target_image_features, recon_image_features = image_features

        total_loss = None  # Initialize as a Tensor
        loss_details = {}
        for name, artifacts in self.loss_functions.items():
            input_type, loss_fn = artifacts
            weight = self.loss_weights[name]
            loss_value = None
            match input_type:
                case "image":
                    loss_value = loss_fn(target_image, recon_image)
                case "image_vgg_features":
                    loss_value = loss_fn(target_image_features, recon_image_features)
                case "latents":
                    loss_value = loss_fn(latents)
                case _:
                    logging.error(
                        f"Loss calculation error for input_type: {input_type}"
                    )
                    continue

            weighted_loss = weight * loss_value
            if not total_loss:
                total_loss = weighted_loss
            else:
                total_loss += weighted_loss
            loss_details[
                name
            ] = loss_value.item()  # Store individual loss values (optional)

        return total_loss, loss_details

    def load_from_config(self, config):
        for loss_config in config:
            name = loss_config["name"]
            weight = loss_config["weight"]
            use = loss_config["use"]
            input_type = None
            if use:
                loss_fn = None
                match name:
                    case "L1 Image loss":
                        loss_fn = F.l1_loss
                        input_type = "image"
                        logging.info(f"Loading loss:- {name}")

                    case "L1 Perceptual loss":
                        loss_fn = F.l1_loss
                        input_type = "image_vgg_features"
                        logging.info(f"Loading loss:- {name}")

                    case "L2 Image loss":
                        loss_fn = F.mse_loss
                        input_type = "image"
                        logging.info(f"Loading loss:- {name}")

                    case "L2 Perceptual loss":
                        loss_fn = F.mse_loss
                        input_type = "image_vgg_features"
                        logging.info(f"Loading loss:- {name}")

                    case "Population K2 Normality Loss":
                        loss_fn = self.compute_sample_k2_loss
                        input_type = "latents"

                    case _:
                        logging.error(f"Unknown loss function:- {name}")
                        continue

                self.register_loss(name, loss_fn, input_type, weight)

    # Define custom losses

    # Enforces normality in generated latents based on D'Agostino Pearson sample normality test
    def compute_sample_k2_loss(sample):
        n = sample.numel()

        # Mean and standard deviation
        mean = sample.mean()
        std_dev = sample.std(unbiased=False)

        # Standardize the sample
        standardized_sample = (sample - mean) / std_dev

        # Compute skewness
        skewness = (standardized_sample**3).mean()

        # Compute kurtosis
        kurtosis = (standardized_sample**4).mean() - 3

        # Compute the K2 statistic as a sum of squares of skewness and kurtosis
        k2_statistic = skewness**2 + kurtosis**2

        return k2_statistic
