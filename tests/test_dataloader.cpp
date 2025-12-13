#include "mnist_dataset.hpp"
#include "dataloader.hpp"
#include <iostream>
#include <cassert>
#include <stdexcept>

int main() {
    try {
        // Assume data is in the current directory or 'mnist' subdirectory
        // The get_mnist.sh script downloads into 'mnist' folder.
        // If we run this from build/, we might need ../mnist or copy it.
        // We'll assume the user runs the test where 'mnist' folder is accessible.
        
        const std::string data_path = "mnist"; 
        
        std::cout << "Initializing MNISTDataset..." << std::endl;
        MNISTDataset dataset(data_path);
        
        std::cout << "Loading data from " << data_path << "..." << std::endl;
        dataset.load();
        
        const Matrix& X_train = dataset.getTrainImages();
        const Matrix& Y_train = dataset.getTrainLabels();
        
        std::cout << "Verifying dimensions..." << std::endl;
        if (X_train.rows != 60000) throw std::runtime_error("Train images count mismatch");
        if (X_train.cols != 784)   throw std::runtime_error("Train images size mismatch");
        if (Y_train.rows != 60000) throw std::runtime_error("Train labels count mismatch");
        if (Y_train.cols != 10)    throw std::runtime_error("Train labels classes mismatch (expected 10)");

        // Check value range (normalization)
        // Check first pixel of first image (might be 0, but check bounds)
        if (X_train(0, 0) < 0.0 || X_train(0, 0) > 1.0) {
             throw std::runtime_error("Pixel value out of range [0, 1]");
        }
        
        std::cout << "Verified Data Loading." << std::endl;
        
        // Test DataLoader
        std::cout << "Testing DataLoader..." << std::endl;
        size_t batch_size = 64;
        DataLoader loader(X_train, Y_train, batch_size);
        
        loader.reset();
        int batches = 0;
        size_t total_samples = 0;
        
        while(loader.hasNext()) {
            auto batch = loader.nextBatch();
            assert(batch.first.rows <= batch_size);
            assert(batch.first.cols == 784);
            assert(batch.second.rows == batch.first.rows);
            assert(batch.second.cols == 10);
            
            total_samples += batch.first.rows;
            batches++;
        }
        
        if (total_samples != 60000) throw std::runtime_error("DataLoader did not yield all samples");
        
        std::cout << "DataLoader iterated " << batches << " batches, total samples: " << total_samples << std::endl;
        std::cout << "Test PASSED." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << std::endl;
        std::cerr << "Make sure you have run 'mnist_reader/get_mnist.sh' and the 'mnist' folder is accessible." << std::endl;
        return 1;
    }
    
    return 0;
}
