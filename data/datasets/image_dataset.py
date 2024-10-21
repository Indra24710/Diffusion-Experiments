import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
import logging


# Dataset for loading images
class ImagesDataset(Dataset):
    def __init__(self, image_dir, data_config, num_files=1):
        self.image_paths = random.sample(
            [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.endswith(data_config["filetype"])
            ],
            num_files,
        )
        logging.info(f"No of files : {num_files}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image_tensor = torch.tensor(
            (np.array(image) / 255.0).astype("float32")
        ).permute(2, 0, 1)
        image_tensor = (image_tensor - 0.5) * 2  # Normalize to [-1, 1]
        return image_tensor, image_path
