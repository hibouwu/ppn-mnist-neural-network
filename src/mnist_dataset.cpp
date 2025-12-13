#include "mnist_dataset.hpp"
// Include the C header from mnist_reader
extern "C" {
    #include "mnist_reader.h"
}
#include <stdexcept>
#include <iostream>
#include <cstdio>

MNISTDataset::MNISTDataset(const std::string& data_dir)
    : data_dir_(data_dir) {}

void MNISTDataset::load() {
    // 60000 training images, 10000 test images
    train_images_ = loadImages(data_dir_ + "/train-images-idx3-ubyte", 60000);
    train_labels_ = loadLabels(data_dir_ + "/train-labels-idx1-ubyte", 60000);
    
    test_images_  = loadImages(data_dir_ + "/t10k-images-idx3-ubyte", 10000);
    test_labels_  = loadLabels(data_dir_ + "/t10k-labels-idx1-ubyte", 10000);

    std::cout << "Loaded MNIST Data:" << std::endl;
    std::cout << "  Train: " << train_images_.rows << " samples." << std::endl;
    std::cout << "  Test:  " << test_images_.rows << " samples." << std::endl;
}

Matrix MNISTDataset::loadImages(const std::string& filename, int count, int size) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    // Use mnist_reader function
    // n=size=28*28=784 (ignored by readMnistImages implementation usually, but strictly it takes N and n)
    // Looking at mnist_reader.c: readMnistImages(FILE* imageFile, int32_t N, int32_t n)
    // N = number of images, n = size of image (bytes?)
    
    // We want "Skip 0, Read count".
    uint8_t* data = readMnistImages(f, 0, count);
    fclose(f);
    
    if (!data) {
        throw std::runtime_error("Failed to read images from " + filename);
    }

    // Convert to Matrix (batch_size, 784) and normalize [0, 1]
    Matrix mat(count, 784);
    for (int i = 0; i < count * 784; ++i) {
        mat.data[i] = static_cast<double>(data[i]) / 255.0;
    }
    
    // Free the raw buffer allocated by readMnistImages (it uses malloc)
    free(data); 

    return mat;
}

Matrix MNISTDataset::loadLabels(const std::string& filename, int count, int num_classes) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    uint8_t* data = readMnistLabels(f, 0, count);
    fclose(f);

    if (!data) {
        throw std::runtime_error("Failed to read labels from " + filename);
    }

    // Convert to One-Hot Matrix (batch_size, num_classes)
    Matrix mat(count, num_classes, 0.0);
    for (int i = 0; i < count; ++i) {
        uint8_t label = data[i];
        if (label < num_classes) {
            mat(i, label) = 1.0;
        }
    }

    free(data);
    return mat;
}
