#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <utility>
#include <vector>
#include <cstddef>
#include "tensor.hpp"

class DataLoader {
public:
    DataLoader(const Matrix& inputs, const Matrix& targets, size_t batchSize);

    void reset();
    bool hasNext() const;
    std::pair<Matrix, Matrix> nextBatch();
    void shuffle(); 

private:
    const Matrix& inputs_;
    const Matrix& targets_;
    size_t batchSize_;
    size_t currentIndex_;
    std::vector<size_t> indices_;
};

#endif // DATALOADER_HPP
