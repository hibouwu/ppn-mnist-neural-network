#include "tensor.hpp"

#include <stdexcept>
#include <iostream>
#include <random>
#include <limits>
#include <cstdlib>   // getenv
#include <cstring>   // strcmp

#include <omp.h>
#include <cblas.h>

#ifdef PROFILE_MATMUL
#include <chrono>
#include "profiling.hpp"
#endif

namespace {

// Choix à l'exécution via la variable d'environnement :
//   MATMUL_IMPL = blas | ijk | ikj | blocked
// Valeur par défaut : blas

enum class MatmulImpl { Blas, Ijk, Ikj, Blocked, Omp };

static inline MatmulImpl parse_impl_env() {
    const char* v = std::getenv("MATMUL_IMPL");
    if (!v || !*v) return MatmulImpl::Blas;

    if (std::strcmp(v, "blas") == 0)    return MatmulImpl::Blas;
    if (std::strcmp(v, "ijk") == 0)     return MatmulImpl::Ijk;
    if (std::strcmp(v, "ikj") == 0)     return MatmulImpl::Ikj;
    if (std::strcmp(v, "blocked") == 0) return MatmulImpl::Blocked;
    if (std::strcmp(v, "omp") == 0)     return MatmulImpl::Omp;

    // Valeur inconnue : avertissement (une seule fois), puis fallback BLAS
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::cerr << "[WARN] MATMUL_IMPL inconnu ('" << v
                  << "'). Utilisation de 'blas'. Valeurs valides : "
                  << "blas | ijk | ikj | blocked | omp\n";
    }
    return MatmulImpl::Blas;
}

// Lecture paresseuse (une seule fois) de la variable d'environnement
static inline MatmulImpl current_impl() {
    static MatmulImpl impl = parse_impl_env();
    return impl;
}

static inline size_t minz(size_t a, size_t b) { return a < b ? a : b; }

// Toutes calculent : C = A * B
// A : MxK, B : KxN, C : MxN
// La matrice C est entièrement écrite (initialisée à zéro ici)

// Version naïve (ordre i-j-k)
static void sgemm_ijk(const Scalar* A, const Scalar* B, Scalar* C,
                      size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Scalar acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = acc;
        }
    }
}

// Version cache-friendly (ordre i-k-j)
static void sgemm_ikj(
#if defined(__GNUC__) || defined(__clang__)
    const Scalar* __restrict A,
    const Scalar* __restrict B,
#else
    const Scalar* A,
    const Scalar* B,
#endif
    Scalar* C,
    size_t M, size_t N, size_t K) {

    // Initialisation de C à zéro
    for (size_t i = 0; i < M; ++i) {
        Scalar* Ci = C + i*N;
        for (size_t j = 0; j < N; ++j) Ci[j] = 0.0f;
    }

    for (size_t i = 0; i < M; ++i) {
        const Scalar* Ai = A + i*K;
        Scalar* Ci = C + i*N;
        for (size_t k = 0; k < K; ++k) {
            const Scalar aik = Ai[k];
            const Scalar* Bk = B + k*N;
            for (size_t j = 0; j < N; ++j) {
                Ci[j] += aik * Bk[j];
            }
        }
    }
}

// Version bloquée (cache blocking)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

static void sgemm_blocked(const Scalar* A, const Scalar* B, Scalar* C,
                          size_t M, size_t N, size_t K) {

    // Initialisation de C à zéro
    for (size_t i = 0; i < M; ++i) {
        Scalar* Ci = C + i*N;
        for (size_t j = 0; j < N; ++j) Ci[j] = 0.0f;
    }

    const size_t BS = static_cast<size_t>(BLOCK_SIZE);

    for (size_t ii = 0; ii < M; ii += BS) {
        const size_t i_max = minz(ii + BS, M);
        for (size_t kk = 0; kk < K; kk += BS) {
            const size_t k_max = minz(kk + BS, K);
            for (size_t jj = 0; jj < N; jj += BS) {
                const size_t j_max = minz(jj + BS, N);
                const size_t len = j_max - jj;

                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        const Scalar aik = A[i*K + k];
                        const Scalar* Bk = B + k*N + jj;
                        Scalar* Ci = C + i*N + jj;

                        for (size_t x = 0; x < len; ++x) {
                            Ci[x] += aik * Bk[x];
                        }
                    }
                }
            }
        }
    }
}

// Version OpenMP (parallélisation de la boucle externe i)
// Basée sur dgemm_ikj pour l'efficacité mémoire
static void sgemm_omp(const Scalar* A, const Scalar* B, Scalar* C,
                      size_t M, size_t N, size_t K) {
    
    // Initialisation de C à zéro (parallèle)
    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        Scalar* Ci = C + i*N;
        for (size_t j = 0; j < N; ++j) {
            Ci[j] = 0.0f;
        }
    }

    // Multiplication (parallèle)
    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        const Scalar* Ai = A + i*K;
        Scalar* Ci = C + i*N;
        for (size_t k = 0; k < K; ++k) {
            const Scalar aik = Ai[k];
            const Scalar* Bk = B + k*N;
            for (size_t j = 0; j < N; ++j) {
                Ci[j] += aik * Bk[j];
            }
        }
    }
}

} // namespace


Matrix::Matrix(size_t r, size_t c) : data(r * c), rows(r), cols(c) {}

Matrix::Matrix(size_t r, size_t c, double init_value)
    : data(r * c, init_value), rows(r), cols(c) {}

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

void Matrix::matmul_into(const Matrix& other, Matrix& out, bool transA, bool transB) const {
    const size_t M = transA ? cols : rows;
    const size_t K = transA ? rows : cols;
    const size_t Kb = transB ? other.cols : other.rows;
    const size_t N = transB ? other.rows : other.cols;

    if (K != Kb) {
        throw std::invalid_argument(
            "Multiplication matricielle impossible : dimensions incompatibles");
    }

    // BLAS utilise des int pour les dimensions
    const size_t max_int = static_cast<size_t>(std::numeric_limits<int>::max());
    if (rows > max_int || cols > max_int ||
        other.rows > max_int || other.cols > max_int ||
        M > max_int || N > max_int || K > max_int) {
        throw std::overflow_error(
            "Dimensions trop grandes pour l'interface BLAS (int)");
    }

    if (out.rows != M || out.cols != N) {
        throw std::invalid_argument(
            "matmul_into: dimensions de sortie incompatibles");
    }


#ifdef PROFILE_MATMUL
    const auto t0 = std::chrono::high_resolution_clock::now();
#endif

   
    const MatmulImpl impl = current_impl();

    if (impl == MatmulImpl::Blas) {
        // Appel BLAS (row-major) avec flags de transposition.
        cblas_sgemm(
            CblasRowMajor,
            transA ? CblasTrans : CblasNoTrans,
            transB ? CblasTrans : CblasNoTrans,
            static_cast<int>(M),
            static_cast<int>(N),
            static_cast<int>(K),
            1.0f,
            data.data(),
            static_cast<int>(cols),
            other.data.data(),
            static_cast<int>(other.cols),
            0.0f,
            out.data.data(),
            static_cast<int>(N)
        );
    } else if (!transA && !transB) {
        // Chemins optimisés existants pour A*B sans transposition logique.
        switch (impl) {
            case MatmulImpl::Ijk:
                sgemm_ijk(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case MatmulImpl::Ikj:
                sgemm_ikj(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case MatmulImpl::Blocked:
                sgemm_blocked(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case MatmulImpl::Omp:
                sgemm_omp(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case MatmulImpl::Blas:
                break;
        }
    } else {
        // Chemin générique pour supporter transA/transB sans copies intermédiaires.
        for (size_t i = 0; i < M; ++i) {
            Scalar* out_row = out.data.data() + i * N;
            for (size_t j = 0; j < N; ++j) {
                Scalar acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    const Scalar a = transA
                        ? data[k * cols + i]
                        : data[i * cols + k];
                    const Scalar b = transB
                        ? other.data[j * other.cols + k]
                        : other.data[k * other.cols + j];
                    acc += a * b;
                }
                out_row[j] = acc;
            }
        }
    }

#ifdef PROFILE_MATMUL
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    const char* name = "blas";
    switch (impl) {
        case MatmulImpl::Blas:    name = "blas";    break;
        case MatmulImpl::Ijk:     name = "ijk";     break;
        case MatmulImpl::Ikj:     name = "ikj";     break;
        case MatmulImpl::Blocked: name = "blocked"; break;
        case MatmulImpl::Omp:     name = "omp";     break;
    }
    
    // Chemin critique : aucune I/O par appel. Statistiques agrégées uniquement.
    matmulProfileRecord(name, us);

#endif
}

Matrix Matrix::matmul(const Matrix& other) const {
    Matrix result(rows, other.cols);
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
