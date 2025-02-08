# Diffusion-Experiments
This repository contains experiments related to diffusion models, focusing on image generation and inversion techniques.

# Requirements

Ensure you have the following installed:

Python (version specified in env/diffusion-env.yml)

Conda

# Installation

Clone the repository:

```
git clone https://github.com/Indra24710/Diffusion-Experiments.git
cd Diffusion-Experiments
```

Install the package and dependencies:

```
pip install -e .
conda env create -f env/diffusion-env.yml
```

Activate the Conda environment:

```
conda activate diffusion-env
```
# Usage
After setting up, you can run various experiments as outlined in the Quickstart section. Modify the configuration files in the config/ directory to tailor the experiments to your needs.

# Quickstart
- **Sample experiment for generating images**: `python .\scripts\start_image_generation.py --config .\config\ldm-celebahq-256.yaml --num_images 10 --expt_name ldm-celebahq-256_expt-1_vanilla-image-generation
- **Sample experiment for inverting images using vanilla ddim inversion**:
  -  Set expt_type to ddim_inversion in config file.
  - `python .\scripts\start_ddim_inversion.py --config .\config\ldm-celebahq-256.yaml --expt_name ldm-celebahq-256_expt-1_vanilla-ddim-inversion`
- **Sample experiment for inverting images using hybrid ddim inversion**:
  -  Set expt_type to hybrid_ddim_inversion in config file.
  - `python .\scripts\start_ddim_inversion.py --config .\config\ldm-celebahq-256.yaml --expt_name ldm-celebahq-256_expt-1_hybrid-ddim-inversion`

# Acknowledgements
- https://github.com/huggingface
- https://huggingface.co/CompVis/ldm-celebahq-256
