# zephyr_dev

## Get Started

### Set up environment

We recommend build the environment and install all required packages using [Anaconda](https://www.anaconda.com/products/individual)

```
conda create -n zephyr python=3.7.5
conda activate zephyr
```

### Install dependencies and compile the C++ module

```
# Required packages for compiling the C++ module
sudo apt-get install build-essential cmake

# Required packages for running the main package
pip install plyfile
conda install -y pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
conda install -y -c conda-forge pytorch-lightning=0.7.6 addict opencv
conda install -y -c open3d-admin open3d
conda install -y scipy scikit-learn scikit-image psutil pandas tensorboard
conda install -y -c conda-forge eigen pcl=1.9.1 xtensor xtensor-python pybind11
```

Compile the c++ library for python bindings in the conda virtual environment

```
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=/location/of/conda/env/bin/python3
make; make install
```

Install the current python package

```
cd .. # Change to the root folder of this project
pip install -e .
```

## Train the network

### Train on YCB-V dataset

These commands will train the network on the real-world images in the YCB-Video training set. 

**On object Set 1 (objects with odd ID)**

```
python train.py \
    --model_name pn2 \
    --dataset_root ./data/ycb/matches_data_train/ \
    --dataset_name ycbv \
    --dataset HSVD_diff_uv_norm \
    --no_valid_proj --no_valid_depth \
    --loss_cutoff log \
    --exp_name final
```

For debugging
```
CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_name pn2 \
    --num_workers 1 \
    --dataset_root ./data/matches_data_small/ \
    --dataset_name ycbv \
    --dataset HSVD_diff_uv_norm \
    --no_valid_proj --no_valid_depth \
    --loss_cutoff log \
    --exp_name final
```

**On object Set 2 (objects with even ID)**

```
python train.py \
    --model_name pn2 \
    --dataset_root ./data/ycb/matches_data_train/ \
    --dataset_name ycbv \
    --dataset HSVD_diff_uv_norm \
    --no_valid_proj --no_valid_depth \
    --loss_cutoff log \
    --val_obj odd \
    --exp_name final_valodd
```

### Train on LM-O synthetic dataset

This command will train the network on the synthetic images provided by [BlenderProc4BOP](https://github.com/DLR-RM/BlenderProc/blob/main/README_BlenderProc4BOP.md). We take the [lm_train_pbr.zip](http://ptak.felk.cvut.cz/6DB/public/bop_datasets/lm_train_pbr.zip) as the training set but the network is only supervised on objects that is in Linemod but not in Linemod-Occluded (i.e. IDs for training objects are `2 3 4 7 13 14 15`). 

```
python train.py \
    --model_name pn2 \
    --dataset_root ./data/bop/lmo/grid_0.7m_train_pbr_match_data/ \
    --dataset_name lmo \
    --dataset HSVD_diff_uv_norm \
    --no_valid_proj --no_valid_depth \
    --loss_cutoff log \
    --exp_name final
```

## Test the network

