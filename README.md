# Latent Diffusion Model Implementation

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
</p>

## About

This is the modified implementation of [pytorch-diffusion](https://github.com/stanipov/pytorch-diffusion) that is integrated into a [pytorch-project-template](https://github.com/Blinorot/pytorch_project_template).

## Installation

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

> [!IMPORTANT]
> Before running training or inference, you need to change the configurations in the following files:
- test folder path in [eval.yaml](src/configs/datasets/eval.yaml) and/or in [train.yaml](src/configs/datasets/train.yaml)
- device type in [fid_metric.yaml](src/configs/metrics/fid_metric.yaml)
- batch-size in [dataloader config](src/configs/dataloader/example.yaml)

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Credits

This repository is based on the [stanipov](https://github.com/stanipov/pytorch-diffusion) and [CompVis](https://github.com/CompVis/latent-diffusion) implementations, [pytorch-project-template](https://github.com/Blinorot/pytorch_project_template) template and [lucidrains diffusion implementation](https://github.com/lucidrains/denoising-diffusion-pytorch) as well as some implementations of [hugginface diffusers](https://github.com/huggingface/diffusers)
