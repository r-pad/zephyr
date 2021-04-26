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
conda install -c conda-forge eigen pcl=1.9.1 xtensor xtensor-python pybind11 opencv
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
pip install -e .
```