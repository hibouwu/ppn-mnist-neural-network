#include "tensor.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

#include <omp.h>

MatrixBuffer::MatrixBuffer(std::size_t n, MatrixInit init)
    : data_(n ? std::make_unique<Scalar[]>(n) : nullptr), size_(n) {
    if (init == MatrixInit::Zero) {
        std::fill(begin(), end(), 0.0f);
    }
}

MatrixBuffer::MatrixBuffer(std::size_t n, Scalar init_value)
    : data_(n ? std::make_unique<Scalar[]>(n) : nullptr), size_(n) {
    std::fill(begin(), end(), init_value);
}

MatrixBuffer::MatrixBuffer(const MatrixBuffer& other)
    : data_(other.size_ ? std::make_unique<Scalar[]>(other.size_) : nullptr),
      size_(other.size_) {
    std::copy(other.begin(), other.end(), begin());
}

MatrixBuffer& MatrixBuffer::operator=(const MatrixBuffer& other) {
    if (this == &other) {
        return *this;
    }

    MatrixBuffer copy(other);
    data_ = std::move(copy.data_);
    size_ = copy.size_;
    return *this;
}

MatrixBuffer& MatrixBuffer::operator=(std::initializer_list<Scalar> values) {
    if (values.size() != size_) {
        throw std::invalid_argument(
            "MatrixBuffer initializer size does not match existing matrix storage");
    }

    std::copy(values.begin(), values.end(), begin());
    return *this;
}


Matrix::Matrix(size_t r, size_t c)
    : data(r * c, MatrixInit::Zero), rows(r), cols(c) {}

Matrix::Matrix(size_t r, size_t c, MatrixInit init)
    : data(r * c, init), rows(r), cols(c) {}

Matrix::Matrix(size_t r, size_t c, double init_value)
    : data(r * c, static_cast<Scalar>(init_value)), rows(r), cols(c) {}

Matrix::Matrix(const Matrix& other)
    : data(other.data), rows(other.rows), cols(other.cols) {}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

Scalar& Matrix::operator()(size_t r, size_t c) {
    if (r >= rows || c >= cols)
        throw std::out_of_range("Index hors limites");
    return data[r * cols + c];
}

const Scalar& Matrix::operator()(size_t r, size_t c) const {
    if (r >= rows || c >= cols)
        throw std::out_of_range("Index hors limites");
    return data[r * cols + c];
}

Matrix Matrix::add(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument(
            "Dimensions incompatibles pour l'addition");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

void Matrix::addInPlace(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument(
            "Dimensions incompatibles pour l'addition inplace");
    }
    const std::size_t n = data.size();
    for (std::size_t i = 0; i < n; ++i) {
        data[i] += other.data[i];
    }
}

Matrix Matrix::mul(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument(
            "Dimensions incompatibles pour la multiplication élément par élément");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

void Matrix::fill(Scalar value) {
    std::fill(data.begin(), data.end(), value);
}

void Matrix::parallelFill(Scalar value) {
    Scalar* ptr = data.data();
    const std::size_t n = data.size();

    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        ptr[i] = value;
    }
}

void Matrix::parallelFillZero() {
    parallelFill(0.0f);
}

Matrix Matrix::matmul(const Matrix& other) const {
    Matrix result(rows, other.cols, MatrixInit::Uninitialized);
    matmul_into(other, result);
    return result;
}

void Matrix::randomInit(double param1, double param2, bool use_normal, unsigned int seed) {
    // If seed is 0, use random_device (random seed)
    // If seed != 0, use fixed seed
    std::mt19937 gen;
    if (seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(seed);
    }

    if (use_normal) {
        // param1 = mean, param2 = stddev
        std::normal_distribution<> dis(param1, param2);
        for (auto& val : data) {
            val = dis(gen);
        }
    } else {
        // param1 = min, param2 = max
        std::uniform_real_distribution<> dis(param1, param2);
        for (auto& val : data) {
            val = dis(gen);
        }
    }
}

void Matrix::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << (*this)(i, j) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
