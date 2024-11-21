import logging
import torch


def init_optimizer(config, tensor):
    match config["inversion"]["optimizer"]["name"]:
        case "adam":
            optimizer = torch.optim.Adam(
                [tensor], **config["inversion"]["optimizer"]["params"]
            )
        case _:
            logging.ERROR(
                f"Unsupported optimizer {config['inversion']['optimizer']['name']}"
            )

    return optimizer
