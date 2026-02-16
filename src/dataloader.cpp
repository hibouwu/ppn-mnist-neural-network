#include "dataloader.hpp"
#include <algorithm>
#include <random>
#include <numeric>
#include <stdexcept>

DataLoader::DataLoader(const Matrix& inputs, const Matrix& targets, size_t batchSize, unsigned int seed)
    : inputs_(inputs),
      targets_(targets),
      batchSize_(batchSize),
      currentIndex_(0),
      rng_(seed == 0 ? std::mt19937(std::random_device{}()) : std::mt19937(seed)) {

    if (inputs_.rows != targets_.rows) {
        throw std::invalid_argument("DataLoader: inputs/targets rows mismatch.");
    }
    if (batchSize_ == 0) {
        throw std::invalid_argument("DataLoader: batch size must be > 0.");
    }
    
    // Initialize indices
    indices_.resize(inputs_.rows);
    std::iota(indices_.begin(), indices_.end(), 0);
}

void DataLoader::reset() {
    currentIndex_ = 0;
    shuffle();
}

void DataLoader::shuffle() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

bool DataLoader::hasNext() const {
    return currentIndex_ < inputs_.rows;
}

size_t DataLoader::nextBatchInto(Matrix& x, Matrix& y) {
    if (!hasNext()) return 0;

    size_t end = std::min(currentIndex_ + batchSize_, inputs_.rows);
    size_t actualSize = end - currentIndex_;

    if (x.rows != actualSize || x.cols != inputs_.cols) {
        x = Matrix(actualSize, inputs_.cols);
    }
    if (y.rows != actualSize || y.cols != targets_.cols) {
        y = Matrix(actualSize, targets_.cols);
    }

    for (size_t i = 0; i < actualSize; ++i) {
        size_t idx = indices_[currentIndex_ + i];
        const double* src_x = inputs_.data.data() + idx * inputs_.cols;
        double* dst_x = x.data.data() + i * inputs_.cols;
        std::copy_n(src_x, inputs_.cols, dst_x);

        const double* src_y = targets_.data.data() + idx * targets_.cols;
        double* dst_y = y.data.data() + i * targets_.cols;
        std::copy_n(src_y, targets_.cols, dst_y);
    }

    currentIndex_ += actualSize;
    return actualSize;
}
