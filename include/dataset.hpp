#pragma once

#include "tensor.hpp"
#include <cstddef>
#include <string>

struct DatasetInfo {
    std::string name;
    std::size_t input_channels = 0;
    std::size_t input_height = 0;
    std::size_t input_width = 0;
    std::size_t input_dim = 0;
    std::size_t num_classes = 0;
};

class Dataset {
public:
    virtual ~Dataset() = default;

    virtual void load() = 0;

    const DatasetInfo& info() const { return info_; }
    const Matrix& getTrainImages() const { return train_images_; }
    const Matrix& getTrainLabels() const { return train_labels_; }
    const Matrix& getTestImages() const { return test_images_; }
    const Matrix& getTestLabels() const { return test_labels_; }

protected:
    DatasetInfo info_;
    Matrix train_images_{0, 0};
    Matrix train_labels_{0, 0};
    Matrix test_images_{0, 0};
    Matrix test_labels_{0, 0};
};
