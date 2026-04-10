#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstddef>  // size_t
#include <memory>
#include <initializer_list>

using Scalar = float;

enum class MatrixInit {
    Zero,
    Uninitialized,
};

class MatrixBuffer {
public:
    MatrixBuffer() = default;
    explicit MatrixBuffer(std::size_t n, MatrixInit init = MatrixInit::Zero);
    MatrixBuffer(std::size_t n, Scalar init_value);

    MatrixBuffer(const MatrixBuffer& other);
    MatrixBuffer& operator=(const MatrixBuffer& other);
    MatrixBuffer(MatrixBuffer&& other) noexcept = default;
    MatrixBuffer& operator=(MatrixBuffer&& other) noexcept = default;

    MatrixBuffer& operator=(std::initializer_list<Scalar> values);

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    Scalar* data() noexcept { return data_.get(); }
    const Scalar* data() const noexcept { return data_.get(); }

    Scalar& operator[](std::size_t idx) noexcept { return data_[idx]; }
    const Scalar& operator[](std::size_t idx) const noexcept { return data_[idx]; }

    Scalar* begin() noexcept { return data_.get(); }
    Scalar* end() noexcept { return data_.get() + size_; }
    const Scalar* begin() const noexcept { return data_.get(); }
    const Scalar* end() const noexcept { return data_.get() + size_; }

private:
    std::unique_ptr<Scalar[]> data_;
    std::size_t size_ = 0;
};

class Matrix {
public:
    // Ordre : data, rows, cols (dans le même ordre que dans les constructeurs)
    MatrixBuffer data;  // 1er
    size_t rows;        // 2ème
    size_t cols;        // 3ème

    // Constructeurs
    Matrix(size_t r, size_t c);
    Matrix(size_t r, size_t c, MatrixInit init);
    Matrix(size_t r, size_t c, double init_value);

    // Constructeur de copie, opérateur d'assignation
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(Matrix&& other) noexcept = default;

    // Destructeur
    ~Matrix() = default;

    // Accès à un élément (ligne, colonne)
    Scalar& operator()(size_t r, size_t c);
    const Scalar& operator()(size_t r, size_t c) const;

    // Opérations de base
    Matrix add(const Matrix& other) const;
    void addInPlace(const Matrix& other);
    Matrix mul(const Matrix& other) const;        // multiplication élément par élément
    Matrix matmul(const Matrix& other) const;     // produit matriciel
    void matmul_into(const Matrix& other, Matrix& out, bool transA = false, bool transB = false) const;
    void fill(Scalar value);
    void parallelFill(Scalar value);
    void parallelFillZero();


    // Initialisation aléatoire
    // use_normal=false -> Uniform(param1, param2)
    // use_normal=true  -> Normal(mean=param1, stddev=param2)
    // seed=0 -> random_device, seed!=0 -> fixed
    void randomInit(double param1 = -1.0, double param2 = 1.0, bool use_normal = false, unsigned int seed = 0);
  
    // Affichage
    void print() const;
};

#endif
