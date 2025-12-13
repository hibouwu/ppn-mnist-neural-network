#!/bin/bash

mkdir -p mnist
cd mnist

# Original download links from Yann LeCun's website seem to have stopped working
# urlbase="http://yann.lecun.com/exdb/mnist"
urlbase="https://github.com/mkolod/MNIST/raw/refs/heads/master"

wget "${urlbase}/train-images-idx3-ubyte.gz"
wget "${urlbase}/train-labels-idx1-ubyte.gz"
wget "${urlbase}/t10k-images-idx3-ubyte.gz"
wget "${urlbase}/t10k-labels-idx1-ubyte.gz"

gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
