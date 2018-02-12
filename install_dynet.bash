# 17 Nov 2017
# By: Prashant Budania
# The script to install Dynet

# Need to change to CUDA 8 because DyNet doesn't work in CUDA 9
# https://github.com/clab/dynet/issues/988
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-8.0 /usr/local/cuda

# Make sure we compile dynet with GPU support
export BACKEND=cuda
pip install --user -r requirements-dynet.txt
