from random import random

import torch
import numpy as np
from scipy import stats
import os
from tqdm import tqdm


def check_normality(inp, tensor_key=None, load_from_npz_file=False, alpha=0.05):
    """
    Checks if the data in a PyTorch tensor is normally distributed using
    Shapiro-Wilk, Anderson-Darling, and Jarque-Bera tests.

    Parameters:
    - data_tensor: PyTorch tensor containing the data samples.
    - alpha: Significance level for the tests (default is 0.05).

    Returns:
    - isNormal: Boolean indicating if the data is normally distributed.
    """

    data_arr = None

    if load_from_npz_file:
        data_arr = np.load(inp)[tensor_key]

    else:
        data_arr = inp

    # Convert to NumPy array
    data = data_arr.flatten()

    # Initialize a list to keep track of normality results
    normality_results = []

    # D'Agostino-Pearson K² Test
    # k2_stat, k2_p = stats.normaltest(data)
    # k2_normal = k2_p > alpha
    # normality_results.append(k2_normal)

    # Jarque-Bera Test
    # jb_stat, jb_p = stats.jarque_bera(data)
    # jb_normal = jb_p > alpha
    # normality_results.append(jb_normal)

    # Anderson-Darling Test
    anderson_result = stats.anderson(data, dist="norm")
    # Use the significance level closest to alpha
    sig_levels = anderson_result.significance_level / 100
    closest_alpha_index = np.argmin(np.abs(sig_levels - alpha))
    crit_value = anderson_result.critical_values[closest_alpha_index]
    anderson_normal = anderson_result.statistic < crit_value
    normality_results.append(anderson_normal)

    # Determine overall normality: data is normal if all tests indicate normality
    isNormal = all(normality_results)

    return isNormal


input_path = "D:\AI projects\Diffusion-Experiments\experiments\inversion\\2024-11-05_ldm-celebahq-256_expt-1_vanilla-hybrid-inversion\latents"

for i in os.listdir(input_path):
    filepath = os.path.join(input_path, i)
    print(
        i,
        " IsNormal: ",
        check_normality(filepath, tensor_key="latents", load_from_npz_file=True),
    )

li = [np.random.randn(1, 3, 64, 64) for i in range(10)]
count = 0
for i in tqdm(range(1000000)):
    random_seed = np.random.randint(1, 1e9)
    np.random.seed(random_seed)
    vector = np.random.randn(1, 3, 64, 64)
    if check_normality(vector):
        count += 1
print(count)


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
    W2 = torch.where(W2 <= 0, torch.tensor(1e-6, dtype=W2.dtype, device=W2.device), W2)

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
        denom == 0, torch.tensor(1e-6, dtype=denom.dtype, device=denom.device), denom
    )

    term2 = torch.sign(denom) * torch.abs((1.0 - 2.0 / A) / torch.abs(denom)) ** (
        1.0 / 3.0
    )
    Z_kurt = (term1 - term2) / torch.sqrt(2.0 / (9.0 * A))

    # Combine the z-scores
    K2 = Z_skew**2 + Z_kurt**2

    # Return the test statistic as the loss
    return K2
