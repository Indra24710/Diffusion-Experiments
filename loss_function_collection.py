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
                        loss_fn = self.compute_population_k2_loss
                        input_type = "latents"

                    case _:
                        logging.error(f"Unknown loss function:- {name}")
                        continue

                self.register_loss(name, loss_fn, input_type, weight)

    # Define custom losses

    # Enforces normality in generated latents based on D'Agostino Pearson population normality test
    def compute_population_k2_loss(self, x):
        n = x.numel()
        if n < 8:
            raise ValueError(
                f"normality_loss requires at least 8 observations; only n={n} observations were given."
            )

        # Convert n to a tensor
        n = torch.tensor(float(n), dtype=x.dtype, device=x.device)

        # Compute mean and standard deviation
        mean = torch.mean(x)
        m2 = torch.mean((x - mean) ** 2)
        std = torch.sqrt(m2)

        # Compute skewness
        m3 = torch.mean((x - mean) ** 3)
        skewness = m3 / (std**3)

        # Skewness test (following the numpy/scipy logic)
        y = skewness * torch.sqrt(((n + 1.0) * (n + 3.0)) / (6.0 * (n - 2.0)))

        beta2_num = 3.0 * (n**2 + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
        beta2_den = (n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0)
        beta2 = beta2_num / beta2_den

        W2 = -1.0 + torch.sqrt(2.0 * (beta2 - 1.0))
        # Ensure W2 is positive to avoid NaNs in log
        W2 = torch.where(
            W2 <= 0, torch.tensor(1e-6, dtype=W2.dtype, device=W2.device), W2
        )

        delta = 1.0 / torch.sqrt(0.5 * torch.log(W2))
        alpha = torch.sqrt(2.0 / (W2 - 1.0))

        # To avoid division by zero, replace zeros with a small number
        y = torch.where(y == 0, torch.tensor(1e-6, dtype=y.dtype, device=y.device), y)

        Z_skew = delta * torch.log(y / alpha + torch.sqrt((y / alpha) ** 2 + 1.0))

        # Compute kurtosis
        m4 = torch.mean((x - mean) ** 4)
        kurtosis = m4 / (std**4)  # This is the non-excess kurtosis (Fisher=False)

        # Kurtosis test (following the numpy/scipy logic)
        E = 3.0 * (n - 1.0) / (n + 1.0)
        var_b2 = (24.0 * n * (n - 2.0) * (n - 3.0)) / (
            (n + 1.0) ** 2 * (n + 3.0) * (n + 5.0)
        )

        x_kurt = (kurtosis - E) / torch.sqrt(var_b2)

        sqrt_beta1_num = 6.0 * (n**2 - 5.0 * n + 2.0)
        sqrt_beta1_den = (n + 7.0) * (n + 9.0)
        sqrt_beta1_inner = (6.0 * (n + 3.0) * (n + 5.0)) / (n * (n - 2.0) * (n - 3.0))
        sqrt_beta1 = (sqrt_beta1_num / sqrt_beta1_den) * torch.sqrt(sqrt_beta1_inner)

        # Ensure sqrt_beta1 is not zero to avoid division by zero
        sqrt_beta1 = torch.where(
            sqrt_beta1 == 0,
            torch.tensor(1e-6, dtype=sqrt_beta1.dtype, device=sqrt_beta1.device),
            sqrt_beta1,
        )

        A = 6.0 + (8.0 / sqrt_beta1) * (
            2.0 / sqrt_beta1 + torch.sqrt(1.0 + 4.0 / (sqrt_beta1**2))
        )

        # Ensure A - 4 is positive to avoid NaNs in sqrt
        A_minus_4 = A - 4.0
        A_minus_4 = torch.where(
            A_minus_4 <= 0,
            torch.tensor(1e-6, dtype=A_minus_4.dtype, device=A_minus_4.device),
            A_minus_4,
        )

        term1 = 1.0 - 2.0 / (9.0 * A)
        denom = 1.0 + x_kurt * torch.sqrt(2.0 / A_minus_4)

        # Handle division by zero or invalid values
        denom = torch.where(
            denom == 0,
            torch.tensor(1e-6, dtype=denom.dtype, device=denom.device),
            denom,
        )

        term2 = torch.sign(denom) * torch.abs((1.0 - 2.0 / A) / torch.abs(denom)) ** (
            1.0 / 3.0
        )
        Z_kurt = (term1 - term2) / torch.sqrt(2.0 / (9.0 * A))

        # Combine the z-scores
        K2 = Z_skew**2 + Z_kurt**2

        # Return the test statistic as the loss
        return K2

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
