# ZePHyR: Zero-shot Pose Hypothesis Rating

[ZePHyR](https://bokorn.github.io/zephyr/) is a zero-shot 6D object pose estimation pipeline. Its core is a learned scoring function that compares the sensor observation to a sparse object rendering of each candidate pose hypothesis. We used PointNet++ as the network structure and trained and tested on YCB-V and LM-O dataset. 

![ZePHyR pipeline animation](images/ZePHyR_Text_Small.gif)


## Get Started

### Set up environment

1. We recommend build the environment and install all required packages using [Anaconda](https://www.anaconda.com/products/individual). 
```
conda env create -n zephyr --file zephyr_env.yml
conda activate zephyr
```

2. Required packages for compiling the C++ module
'''
sudo apt-get install build-essential cmake
'''

3. Compile the c++ library for python bindings in the conda virtual environment
```
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=/location/of/conda/env/bin/python3
make; make install
```

4. Install the current python package
```
cd .. # Change to the root folder of this project
pip install -e .
```

## Download pre-processed dataset

Download pre-processed data from this Google Drive [link](https://drive.google.com/file/d/1BolVjGJGZIyJ1kW-8PQx2dTjWgWXfkmi/view?usp=sharing) and unzip it in the `python/zephyr/data` folder. The unzipped data takes around 66GB of storage. 

The following commands need to be run in `python/zephyr/` folder. 
```
cd python/zephyr/
```

## Test the network

### Test on YCB-V dataset

Test on the YCB-V dataset using the model trained on objects with odd ID
```
python test.py \
    --model_name pn2 \
    --dataset_root ./data/ycb/matches_data_test/ \
    --dataset_name ycbv \
    --dataset HSVD_diff_uv_norm \
    --no_valid_proj --no_valid_depth \
    --loss_cutoff log \
    --exp_name final \
    --resume_path ./ckpts/final_ycbv.ckpt
```
Test on the YCB-V dataset using the model trained on objects with even ID
```
python test.py \
    --model_name pn2 \
    --dataset_root ./data/ycb/matches_data_test/ \
    --dataset_name ycbv \
    --dataset HSVD_diff_uv_norm \
    --no_valid_proj --no_valid_depth \
    --loss_cutoff log \
    --exp_name final \
    --resume_path ./ckpts/final_ycbv_valodd.ckpt
```

### Test on LM-O dataset

```
python test.py \
    --model_name pn2 \
    --dataset_root ./data/lmo/matches_data_test/ \
    --dataset_name lmo \
    --dataset HSVD_diff_uv_norm \
    --no_valid_proj --no_valid_depth \
    --loss_cutoff log \
    --exp_name final \
    --resume_path ./ckpts/final_lmo.ckpt
```

The testing results will be stored in `test_logs` and the results in BOP Challenge format will be in `test_logs/bop_results`. Please refer to [bop_toolkit](https://github.com/thodan/bop_toolkit) for converting the results to BOP Average Recall scores used in [BOP challenge](https://bop.felk.cvut.cz/home/). 

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
    --dataset_root ./data/lmo/matches_data_train/ \
    --dataset_name lmo \
    --dataset HSVD_diff_uv_norm \
    --no_valid_proj --no_valid_depth \
    --loss_cutoff log \
    --exp_name final
```

# Cite

If you find this codebase useful in your research, please consider citing:
```
@inproceedings{icra2021zephyr,
    title={ZePHyR: Zero-shot Pose Hypothesis Rating},
    author={Brian Okorn, Qiao Gu, Martial Hebert, David Held},
    booktitle={2021 International Conference on Robotics and Automation (ICRA)},
    year={2021}
}
```

# Reference

* We used the [PPF](http://campar.in.tum.de/pub/drost2010CVPR/drost2010CVPR.pdf) implementation provided in [MVTec HALCON](https://www.mvtec.com/products/halcon) software. It is a commercial software but provides [free license for student](https://www.mvtec.com/company/mvtec-on-campus/licenses/student). 