#include "tensor.hpp"
#include <stdexcept>
#include <iostream>
#include <random>

Tensor::Tensor(size_t r, size_t c)
    : rows(r), cols(c), data(r * c) {}

Tensor::Tensor(size_t r, size_t c, double init_value)
    : rows(r), cols(c), data(r * c, init_value) {}

Tensor::Tensor(const Tensor& other)
    : rows(other.rows), cols(other.cols), data(other.data) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

double& Tensor::operator()(size_t r, size_t c) {
    if (r >= rows || c >= cols)
        throw std::out_of_range("Index out of bounds");
    return data[r * cols + c];
}

const double& Tensor::operator()(size_t r, size_t c) const {
    if (r >= rows || c >= cols)
        throw std::out_of_range("Index out of bounds");
    return data[r * cols + c];
}

Tensor Tensor::add(const Tensor& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    Tensor result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::mul(const Tensor& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");
    }
    Tensor result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Cannot multiply tensors: incompatible dimensions");
    }
    Tensor result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Tensor Tensor::transpose() const {
    Tensor result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

void Tensor::randomInit(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    for (auto& val : data) {
        val = dis(gen);
    }
}

void Tensor::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << (*this)(i, j) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

