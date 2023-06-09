# Categorical-3D_SRPE: Diffusion-Driven Self-Supervised Network for Multi-Object 3D Shape Reconstruction and Categorical 6-DoF Pose Estimation
(code will be update......)
# Overview
#### This repository contains the PyTorch implementation of the paper "Diffusion-Driven Self-Supervised Network for Multi-Object 3D Shape Reconstruction and Categorical 6-DoF Pose Estimation". we introduce a novel pretrain-to-reinforce paradigm that contributes to self-supervised categorical 6-DoF pose estimation and 3D shape reconstruction for multiple objects.

![image](https://github.com/S-JingTao/Categorical-3D_SRPE/assets/26479294/bd3a10b6-5b61-469a-8139-138584256075)

# Visualization of the diffusion process in our method
![generate-more-small](https://github.com/S-JingTao/Categorical-3D_SRPE/assets/26479294/03555ae2-e2e0-4c60-a982-f1e3a1bc50bf)

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
Build PointNet++
```bash
cd Categorical-3D_SRPE/lib/pointnet2/pointnet2
python setup.py install
```
Build nn_distance
```bash
cd Categorical-3D_SRPE/lib/nn_distance
python setup.py install
```
# Dataset
Download camera_train, camera_val, real_train, real_test, ground-truth annotations and mesh models provided by NOCS. Then, organize and preprocess these files following SPD https://github.com/mentian/object-deformnet.

Run python scripts to prepare the datasets.
```bash
cd Categorical-3D_SRPE/dataset
python generate_list.py
python shape_data.py
python pose_data.py
```
# Training
train a pre-trained shape reconstruction model:
```bash
cd Categorical-3D_SRPE
python net_train.py
```
fine-tuning model
```bash
python net_fine_tuning.py
```
# Evaluation
evaluate the 6D pose and 3D size
```bash
python evaluate_pose.py
```
evaluate the 3D shape reconstruction
```bash
python evaluate_shape.py
```
# Acknowledgement
Our implementation leverages the code from SPD, SGPA and  NOCS.
