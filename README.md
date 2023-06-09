# Categorical-3D_SRPE: Diffusion-Driven Self-Supervised Network for Multi-Object 3D Shape Reconstruction and Categorical 6-DoF Pose Estimation
# Overview
#### This repository contains the PyTorch implementation of the paper "Diffusion-Driven Self-Supervised Network for Multi-Object 3D Shape Reconstruction and Categorical 6-DoF Pose Estimation". we introduce a novel pretrain-to-reinforce paradigm that contributes to self-supervised categorical 6-DoF pose estimation and 3D shape reconstruction for multiple objects.

![image](https://github.com/S-JingTao/Categorical-3D_SRPE/assets/26479294/bd3a10b6-5b61-469a-8139-138584256075)
# Dependencies
* Python 3.8
* PyTorch 1.0.1
* CUDA 9.0
* PIL
* scipy
* numpy

# Installation
Using the conda to setup the virtual environment. If you have already installed conda, please use the following commands.

```bash
cd $Categorical-3D_SRPE
conda create -y --prefix ./env python=3.8
conda activate ./env/
./env/bin/python -m pip install --upgrade pip
```


cd $ROOT/lib/nn_distance

python setup.py install --user
