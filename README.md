# ppn-mnist-neural-network
Implementation of a neural network from scratch in C++ for the MNIST dataset (PPN project, M1 CHPS)

## Séparation des phrase

[Docs/demande_fr.md](Docs/demande_fr.md)

## Diagramme de classe

- La partie des phases de propagation avant et arrière.

![phase3](Docs/Images/phase3.png)

- La partie entraînement.

![phase4-6](Docs/Images/phase4-6.png)

//build

rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
             


//télécharger des image dans le floder minist

rm -rf mnist
mkdir mnist
cd mnist

wget -O train-images-idx3-ubyte.gz  https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget -O train-labels-idx1-ubyte.gz  https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget -O t10k-images-idx3-ubyte.gz   https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget -O t10k-labels-idx1-ubyte.gz   https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
ls -lh
