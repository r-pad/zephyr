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
conda install -y scipy scikit-learn scikit-image psutil
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