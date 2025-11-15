#include "tensor.hpp"
#include <stdexcept>
#include <iostream>
#include <random>

Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c) {}

Matrix::Matrix(size_t r, size_t c, double init_value) : rows(r), cols(c), data(r * c, init_value) {}

Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

double& Matrix::operator()(size_t r, size_t c) {
    if (r >= rows || c >= cols) throw std::out_of_range("Index out of bounds");
    return data[r * cols + c];
}

const double& Matrix::operator()(size_t r, size_t c) const {
    if (r >= rows || c >= cols) throw std::out_of_range("Index out of bounds");
    return data[r * cols + c];
}

Matrix Matrix::add(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::mul(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Matrix Matrix::matmul(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Cannot multiply matrices: incompatible dimensions");
    }
    Matrix result(rows, other.cols);
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

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

void Matrix::randomInit(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    for (auto& val : data) {
        val = dis(gen);
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
