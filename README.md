# ZePHyR: Zero-shot Pose Hypothesis Rating

[ZePHyR](https://bokorn.github.io/zephyr/) is a zero-shot 6D object pose estimation pipeline. The core is a learned scoring function that compares the sensor observation to a sparse object rendering of each candidate pose hypothesis. We used PointNet++ as the network structure and trained and tested on YCB-V and LM-O dataset. 

[[ArXiv]](https://arxiv.org/abs/2104.13526) [[Project Page]](https://bokorn.github.io/zephyr/) [[Video]](https://www.youtube.com/watch?v=41bxU7U2VZ4) [[BibTex]](https://github.com/r-pad/zephyr#cite)

![ZePHyR pipeline animation](images/ZePHyR_Text_Small.gif)


## Get Started

First, checkout this repo by
```
git clone --recurse-submodules git@github.com:r-pad/zephyr.git
```

### Set up environment

1. We recommend building the environment and install all required packages using [Anaconda](https://www.anaconda.com/products/individual). 
```
conda env create -n zephyr --file zephyr_env.yml
conda activate zephyr
```

2. Install the required packages for compiling the C++ module

```
sudo apt-get install build-essential cmake libopencv-dev python-numpy
```

3. Compile the c++ library for python bindings in the conda virtual environment
```
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=$(python -c "import sys; print(sys.executable)") -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
make; make install
```

4. Install the current python package
```
cd .. # move to the root folder of this repo
pip install -e .
```

## Download pre-processed dataset

Download pre-processed training and testing data (`ycbv_preprocessed.zip`, `lmo_preprocessed.zip` and `ppf_hypos.zip`) from this Google Drive [link](https://drive.google.com/drive/folders/1Fq-24RAnn0uWAEVauwMZgpqtWYkZCqQI?usp=sharing) and unzip it in the `python/zephyr/data` folder. The unzipped data takes around 66GB of storage in total. 

The following commands need to be run in `python/zephyr/` folder. 
```
cd python/zephyr/
```

### Example script to run the network

To use the network, an example is provided in [notebooks/TestExample.ipynb](https://github.com/r-pad/zephyr/blob/main/notebooks/TestExample.ipynb). In the example script, a datapoint is loaded from LM-O dataset provided by the [BOP Challenge](https://bop.felk.cvut.cz/datasets/). The pose hypotheses is provided by PPF algorithm (extracted from `ppf_hypos.zip`). Despite the complex dataloading code, only the following data of the observation and the model point clouds is needed to run the network: 
* `img`: RGB image, np.ndarray of size (H, W, 3) in np.uint8
* `depth`: depth map, np.ndarray of size (H, W) in np.float, in meters
* `cam_K`: camera intrinsic matrix, np.ndarray of size (3, 3) in np.float
* `model_colors`: colors of model point cloud, np.ndarray of size (N, 3) in float, scaled in [0, 1]
* `model_points`: xyz coordinates of model point cloud, np.ndarray of size (N, 3) in float, in meters
* `model_normals`: normal vectors of mdoel point cloud, np.ndarray of size (N, 3) in float, each L2 normalized
* `pose_hypos`: pose hypotheses in camera frame, np.ndarray of size (K, 4, 4) in float

### Run PPF algorithm using HALCON software

The PPF algorithm we used is the [surface matching function](https://www.mvtec.com/doc/halcon/13/en/find_surface_model.html) implmemented in [MVTec HALCON](https://www.mvtec.com/products/halcon/?pk_campaign=EN-Halcon&pk_medium=cpc&pk_kwd=) software. HALCON provides a [Python interface](https://pypi.org/project/mvtec-halcon/) for programmers together with its newest versions. I wrote a simple wrapper which calls `create_surface_model()` and `find_surface_model()` to get the pose hypotheses. See [notebooks/TestExample.ipynb](https://github.com/r-pad/zephyr/blob/main/notebooks/TestExample.ipynb) for how to use it. 

The wrapper requires the HALCON 21.05 to be installed, which is a commercial software but it provides [free licenses for students](https://www.mvtec.com/company/mvtec-on-campus/licenses/student). 

If you don't have access to HALCON, sets of pre-estimated pose hypotheses are provided in the pre-processed dataset. 

## Test the network

Download the pretrained pytorch model checkpoint from this Google Drive [link](https://drive.google.com/file/d/1cBLzDq71peadG5zkJsdQXpJ45coF5HEW/view?usp=sharing) and unzip it in the `python/zephyr/ckpts/` folder.  We provide 3 checkpoints, two trained on YCB-V objects with odd ID (`final_ycbv.ckpt`) and even ID (`final_ycbv_valodd.ckpt`) respectively, and one trained on LM objects that are not in LM-O dataset (`final_lmo.ckpt`). 

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
@inproceedings{okorn2021zephyr,
  title={Zephyr: Zero-shot pose hypothesis rating},
  author={Okorn, Brian and Gu, Qiao and Hebert, Martial and Held, David},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={14141--14148},
  year={2021},
  organization={IEEE}
}
```

# Reference

* We used the [PPF](http://campar.in.tum.de/pub/drost2010CVPR/drost2010CVPR.pdf) implementation provided in [MVTec HALCON](https://www.mvtec.com/products/halcon) software for pose hypothese generation. It is a commercial software but provides [free license for student](https://www.mvtec.com/company/mvtec-on-campus/licenses/student). 
* We used [bop_toolkit](https://github.com/thodan/bop_toolkit) for data loading and results evaluation. 
