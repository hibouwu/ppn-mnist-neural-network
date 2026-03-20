#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include "batch_source.hpp"
#include <vector>
#include <cstddef>
#include <random>
#include <memory>

class DataLoader {
public:
    DataLoader(const Matrix& inputs, const Matrix& targets, size_t batchSize, unsigned int seed = 0);
    DataLoader(std::shared_ptr<const BatchSource> source, size_t batchSize, unsigned int seed = 0);

    void reset();
    bool hasNext() const;
    size_t nextBatchInto(Matrix& x, Matrix& y);
    void shuffle(); 

    size_t batchSize() const { return batchSize_; }
    size_t inputCols() const { return source_->inputCols(); }
    size_t targetCols() const { return source_->targetCols(); }

private:
    std::shared_ptr<const BatchSource> source_;
    size_t batchSize_;
    size_t currentIndex_;
    std::vector<size_t> indices_;
    std::mt19937 rng_;
};

#endif // DATALOADER_HPP
