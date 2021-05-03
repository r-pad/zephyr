# Install python dependencies step-by-step

Create an empty conda environment

```
conda create -n zephyr python=3.7.5
conda activate zephyr
```

`opencv-python-headless` is only needed to fix a [bug](https://github.com/opencv/opencv/issues/5150)
```
pip install plyfile
conda install -y pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
conda install -y -c conda-forge pytorch-lightning=0.7.6 
conda install -y -c open3d-admin open3d
conda install -y scipy scikit-learn scikit-image psutil pandas
conda install -y -c conda-forge eigen pcl=1.9.1 xtensor xtensor-python pybind11
conda install -y -c conda-forge tensorboard addict opencv
conda install -c fastai opencv-python-headless
```