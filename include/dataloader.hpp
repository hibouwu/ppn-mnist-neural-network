#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <vector>
#include <cstddef>
#include "tensor.hpp"

class DataLoader {
public:
    DataLoader(const Matrix& inputs, const Matrix& targets, size_t batchSize, unsigned int seed = 0);

    void reset();
    bool hasNext() const;
    size_t nextBatchInto(Matrix& x, Matrix& y);
    void shuffle(); 

    size_t batchSize() const { return batchSize_; }
    size_t inputCols() const { return inputs_.cols; }
    size_t targetCols() const { return targets_.cols; }

private:
    const Matrix& inputs_;
    const Matrix& targets_;
    size_t batchSize_;
    size_t currentIndex_;
    std::vector<size_t> indices_;
    unsigned int seed_;
};

#endif // DATALOADER_HPP
