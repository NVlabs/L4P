#!/bin/bash

# install miniconda for python environment
mkdir -p /workspace/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/miniconda3/miniconda.sh
bash /workspace/miniconda3/miniconda.sh -b -u -p /workspace/miniconda3
rm -rf /workspace/miniconda3/miniconda.sh
export PATH="${PATH}:/workspace/miniconda3/bin"
sh -c "/workspace/miniconda3/bin/conda init bash" || true

# accept terms and conditions
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# install l4p environment
source /workspace/miniconda3/etc/profile.d/conda.sh
conda init
conda create --name l4p python=3.10 -y
conda activate l4p
pip install -r requirements.txt