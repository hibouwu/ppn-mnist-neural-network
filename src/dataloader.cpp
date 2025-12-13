#include "dataloader.hpp"
#include <algorithm>
#include <random>
#include <numeric>

DataLoader::DataLoader(const Matrix& inputs, const Matrix& targets, size_t batchSize)
    : inputs_(inputs), targets_(targets), batchSize_(batchSize), currentIndex_(0) {
    
    // Initialize indices
    indices_.resize(inputs_.rows);
    std::iota(indices_.begin(), indices_.end(), 0);
}

void DataLoader::reset() {
    currentIndex_ = 0;
    shuffle();
}

void DataLoader::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices_.begin(), indices_.end(), g);
}

bool DataLoader::hasNext() const {
    return currentIndex_ < inputs_.rows;
}

std::pair<Matrix, Matrix> DataLoader::nextBatch() {
    size_t end = std::min(currentIndex_ + batchSize_, inputs_.rows);
    size_t actualSize = end - currentIndex_;

    Matrix x(actualSize, inputs_.cols);
    Matrix y(actualSize, targets_.cols);

    for (size_t i = 0; i < actualSize; ++i) {
        size_t idx = indices_[currentIndex_ + i];
        
        // Copy row idx from inputs to row i of x
        // Assumes flat data layout in Matrix
        for (size_t c = 0; c < inputs_.cols; ++c) {
            x(i, c) = inputs_(idx, c);
        }
        for (size_t c = 0; c < targets_.cols; ++c) {
            y(i, c) = targets_(idx, c);
        }
    }

    currentIndex_ += actualSize;
    return {x, y};
}
