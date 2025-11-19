# ppn-mnist-neural-network
Implementation of a neural network from scratch in C++ for the MNIST dataset (PPN project, M1 CHPS)


//build

rm -rf build          
mkdir build          
cd build              
cmake ..             
make -j                


//test après build

./build/test_autodiff
