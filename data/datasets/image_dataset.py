import os
from torch.utils.data import Dataset
from PIL import Image
import random
import logging
import torchvision.transforms as T


# Dataset for loading images
class ImagesDataset(Dataset):
    def __init__(self, image_dir, data_config, num_files=1):
        # Load all images if num_files in config is equal to no of images in input folder
        if len(os.listdir(image_dir)) == num_files:
            self.image_paths = [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.endswith(data_config["filetype"])
            ]
        else:
            # Randomly samply 'num_files' no of images from the input_folder
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
        transform = T.Compose(
            [
                T.Resize((256, 256)),  # Resize the image to 256x256 pixels
                T.ToTensor(),  # Converts to tensor and normalizes to [0, 1]
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalizes to [-1, 1]
            ]
        )
        image_tensor = transform(image)
        return image_tensor, image_path
