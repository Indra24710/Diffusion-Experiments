import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Setup data for fake image detectors")
    parser.add_argument(
        "--real_image", type=str, required=True, help="Path to the real images folder"
    )
    parser.add_argument(
        "--fake_images", type=str, required=True, help="Path to fake images folder"
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Path to the output folder"
    )
    parser.add_argument(
        "--detector",
        type=str,
        required=True,
        help="which detector to use. Ex:- ufdetect",
    )
    return parser.parse_args()


def universal_fake_detector_data(
    output_folder, input_real_images_folder, input_inverted_images_folder
):
    # create output folder
    os.makedirs(os.path.join(output_folder, "0_real"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "ddim_inverted"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "hyb_ddim_inverted"), exist_ok=True)

    for inv_image in os.listdir(input_inverted_images_folder):
        real_image_filename = inv_image.split("_")[2] + ".png"
        shutil.copy(
            os.path.join(input_real_images_folder, real_image_filename),
            os.path.join(output_folder, "0_real", real_image_filename),
        )
        if "ddim-inv" in inv_image:
            shutil.copy(
                os.path.join(input_inverted_images_folder, inv_image),
                os.path.join(output_folder, "ddim_inverted", inv_image),
            )
        else:
            shutil.copy(
                os.path.join(input_inverted_images_folder, inv_image),
                os.path.join(output_folder, "hyb_ddim_inverted", inv_image),
            )


args = parse_args()
output_folder = args.output_folder
input_real_images_folder = args.real_images
input_inverted_images_folder = args.fake_images
detector = args.detector

match detector:
    case "ufdetect":
        universal_fake_detector_data(
            output_folder, input_real_images_folder, input_inverted_images_folder
        )

    case _:
        print("Invalid detector. Please choose between ufdetector")
