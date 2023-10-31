# FAIT: A Holistic Functionalization Approach to Optimizing Imperative Tensor Programs in Deep Learning 
**TensorSSA** and **For Loop** Auto Parallel in long-tail.
The architecture overview of FAIT is as follows:

![arch_overview](docs/imgs/arch_overview.png)

## Dependency
- LibTorch
- LibTorchVision
## Build From Source
### Linux
```shell
# install torch vision
git clone https://github.com/pytorch/vision.git
cd vision
git checkout release/2.0
mkdir build && cd build
cmake -DWITH_CUDA=ON .. && make && make install

# build pytorch from source
git clone https://github.com/pytorch/pytorch.git --recursive
git checkout v2.0.0
cd pytorch
python setup.py develop --user

# build source
git clone https://github.com/JimyMa/fait.git --recursive
cd fait
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)' ` ..
make -j{$nproc}
```

## Run the imperative tensor program using FAIT
We take the post-process of SSD as an example.
```shell
cd $PATH/OF/FAIT
# Step 1: Download features of SSD extracted by computer vision networks
mkdir -p feats; cd feats
https://github.com/JimyMa/FAIT/releases/download/V0.0.1/ssd_feat.pt

# Step 2: generate scripted graph of imperative tensor program
cd ../models
python ssd_bbox.py

# Step 3: run fait
cd ../build
./fait ../models/ssd_bbox.pt ../models/ssd_bbox.json ../feats/ssd_feat.pt
# Latency: 942.5us

# Step 4 (optional): run TorchScript nvfuser
./run_ts ../models/ssd_bbox.pt ../models/ssd_bbox.json ../feats/ssd_feat.pt
# Latency: 7.386ms (nvfuser backend)
# Latency: 2.053ms (nnc backend)

```




