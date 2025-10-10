# LatentDEM: Blind Inversion using Latent Diffusion Priors

[![Paper](https://img.shields.io/badge/arXiv-2407.01027-b31b1b.svg)](https://arxiv.org/abs/2407.01027)

This repository provides the official `PyTorch` implementation for the paper **Blind Inversion using Latent Diffusion Priors**.

LatentDEM is the first framework to solve blind inverse problems by leveraging pre-trained *latent* diffusion models. All experiments presented were performed using Stable Diffusion v1.5.

## ⚙️ Getting Started

Our implementation is built directly upon the [PSLD repository](https://github.com/LituRout/PSLD). Follow these steps to set up the environment.

### 1. Clone the Repository
Clone this repository, which includes the core PSLD framework and our modifications.
```bash
git clone https://github.com/ai4imaging/latentdem.git
cd latentdem
```

### 2. Set Up the Environment
We provide a Conda environment file. Use it to install the required dependencies.
```bash
# Navigate to the stable-diffusion directory
cd stable-diffusion
# Create and activate the conda environment
conda env create -f environment.yml
conda activate ldm
```
Note: Before proceeding, ensure that the base samplers for both [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) and Stable Diffusion are functioning correctly as per the original PSLD instructions.

### 3. Download Pre-trained Model
Download the Stable Diffusion v1.5 checkpoint into the `stable-diffusion` directory and put it into '/stable-diffusion/models/ldm/stable-diffusion-v1/'
```bash
wget [https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt)
```

## 🚀 Running Experiments

All experiment scripts are designed to be run from within the stable-diffusion directory.

Important: Before executing, please edit the shell scripts in `stable-diffusion/runforlatentDEM/` to set the outdir variable to your desired output path.

### Blind Deblurring

To reproduce the blind deblurring results from the paper, execute the following commands:
```bash
sh runforlatentDEM/inverse_mb_girl.sh
sh runforlatentDEM/inverse_mb_Macaron.sh
sh runforlatentDEM/inverse_mb_man.sh
sh runforlatentDEM/inverse_mb_plant.sh
```

## 📂 Repository Structure

1. `stable-diffusion/runforlatentDEM/`: Contains all executable shell scripts for reproducing experiments.

2. `diffusion-posterior-sampling/dataforlatentDEM/`: Contains the image data used for the paper's experiments.


## Citation
If you find our work useful in your research, please consider citing:
```
@article{bai2024blind,
  title={Blind inversion using latent diffusion priors},
  author={Bai, Weimin and Chen, Siyi and Chen, Wenzheng and Sun, He},
  journal={arXiv preprint arXiv:2407.01027},
  year={2024}
}
```