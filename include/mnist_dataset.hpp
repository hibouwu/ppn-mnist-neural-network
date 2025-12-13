#pragma once

#include "tensor.hpp"
#include <string>

/**
 * @brief Handles loading MNIST data from IDX files using mnist_reader.
 */
class MNISTDataset {
public:
    /**
     * @param data_dir Directory containing the 'mnist' folder or the files.
     */
    explicit MNISTDataset(const std::string& data_dir);

    /**
     * @brief Loads training and test data from files.
     * throws std::runtime_error if files missing or parse error.
     */
    void load();

    const Matrix& getTrainImages() const { return train_images_; }
    const Matrix& getTrainLabels() const { return train_labels_; }
    const Matrix& getTestImages()  const { return test_images_; }
    const Matrix& getTestLabels()  const { return test_labels_; }

private:
    std::string data_dir_;
    Matrix train_images_{0, 0};
    Matrix train_labels_{0, 0};
    Matrix test_images_{0, 0};
    Matrix test_labels_{0, 0};

    // Helper to call C function and convert to Matrix
    Matrix loadImages(const std::string& filename, int count, int size=784);
    Matrix loadLabels(const std::string& filename, int count, int num_classes=10);
};
