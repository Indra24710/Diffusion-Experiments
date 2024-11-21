# Diffusion-Experiments

### Quickstart
- `pip install -e .`
- `conda env create -f env/diffusion-env.yml`
- **Sample experiment for generating images**: `python .\scripts\start_image_generation.py --config .\config\ldm-celebahq-256.yaml --num_images 10 --expt_name ldm-celebahq-256_expt-1_vanilla-image-generation
- **Sample experiment for inverting images using vanilla ddim inversion**:
  -  Set expt_type to ddim_inversion in config file.
  - `python .\scripts\start_ddim_inversion.py --config .\config\ldm-celebahq-256.yaml --expt_name ldm-celebahq-256_expt-1_vanilla-ddim-inversion
- **Sample experiment for inverting images using hybrid ddim inversion**:
  -  Set expt_type to hybrid_ddim_inversion in config file.
  - `python .\scripts\start_ddim_inversion.py --config .\config\ldm-celebahq-256.yaml --expt_name ldm-celebahq-256_expt-1_hybrid-ddim-inversion
