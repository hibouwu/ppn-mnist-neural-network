#include "dataloader.hpp"
#include <algorithm>
#include <random>
#include <numeric>
#include <stdexcept>

namespace {

class MatrixBatchSource : public BatchSource {
public:
    MatrixBatchSource(const Matrix& inputs, const Matrix& targets)
        : inputs_(&inputs), targets_(&targets) {
        if (inputs_->rows != targets_->rows) {
            throw std::invalid_argument("DataLoader: inputs/targets rows mismatch.");
        }
    }

    std::size_t rowCount() const override { return inputs_->rows; }
    std::size_t inputCols() const override { return inputs_->cols; }
    std::size_t targetCols() const override { return targets_->cols; }

    void loadRows(const std::vector<std::size_t>& indices,
                  Matrix& inputs,
                  Matrix& targets) const override {
        if (inputs.rows != indices.size() || inputs.cols != inputs_->cols) {
            inputs = Matrix(indices.size(), inputs_->cols);
        }
        if (targets.rows != indices.size() || targets.cols != targets_->cols) {
            targets = Matrix(indices.size(), targets_->cols);
        }

        for (std::size_t i = 0; i < indices.size(); ++i) {
            const std::size_t idx = indices[i];
            const double* src_x = inputs_->data.data() + idx * inputs_->cols;
            double* dst_x = inputs.data.data() + i * inputs_->cols;
            std::copy_n(src_x, inputs_->cols, dst_x);

            const double* src_y = targets_->data.data() + idx * targets_->cols;
            double* dst_y = targets.data.data() + i * targets_->cols;
            std::copy_n(src_y, targets_->cols, dst_y);
        }
    }

private:
    const Matrix* inputs_;
    const Matrix* targets_;
};

} // namespace

DataLoader::DataLoader(const Matrix& inputs, const Matrix& targets, size_t batchSize, unsigned int seed)
    : DataLoader(std::make_shared<MatrixBatchSource>(inputs, targets), batchSize, seed) {}

DataLoader::DataLoader(std::shared_ptr<const BatchSource> source, size_t batchSize, unsigned int seed)
    : source_(std::move(source)),
      batchSize_(batchSize),
      currentIndex_(0),
      rng_(seed == 0 ? std::mt19937(std::random_device{}()) : std::mt19937(seed)) {
    if (!source_) {
        throw std::invalid_argument("DataLoader: batch source must not be null.");
    }
    if (batchSize_ == 0) {
        throw std::invalid_argument("DataLoader: batch size must be > 0.");
    }
    
    // Initialize indices
    indices_.resize(source_->rowCount());
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
    return currentIndex_ < indices_.size();
}

size_t DataLoader::nextBatchInto(Matrix& x, Matrix& y) {
    if (!hasNext()) return 0;

    size_t end = std::min(currentIndex_ + batchSize_, indices_.size());
    size_t actualSize = end - currentIndex_;
    std::vector<std::size_t> batch_indices(actualSize);
    std::copy(indices_.begin() + static_cast<std::ptrdiff_t>(currentIndex_),
              indices_.begin() + static_cast<std::ptrdiff_t>(end),
              batch_indices.begin());
    source_->loadRows(batch_indices, x, y);

    currentIndex_ += actualSize;
    return actualSize;
}
