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
mkdir build          
cd build              
cmake ..             
make -j                


//test après build

./build/test_autodiff
