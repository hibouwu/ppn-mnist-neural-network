#include "gemm/matmul_internal.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <cblas.h>

#ifdef PROFILE_MATMUL
#include <chrono>

#include "profiling.hpp"
#endif

namespace gemm {

size_t minz(size_t a, size_t b) {
    return a < b ? a : b;
}

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

MatmulImpl parse_impl_env() {
    const char* v = std::getenv("MATMUL_IMPL");
    if (!v || !*v) {
        return MatmulImpl::Blas;
    }

    if (std::strcmp(v, "blas") == 0) return MatmulImpl::Blas;
    if (std::strcmp(v, "ijk") == 0) return MatmulImpl::Ijk;
    if (std::strcmp(v, "ikj") == 0) return MatmulImpl::Ikj;
    if (std::strcmp(v, "blocked") == 0) return MatmulImpl::Blocked;
    if (std::strcmp(v, "omp") == 0) return MatmulImpl::Omp;
    if (std::strcmp(v, "omp_blocked") == 0) return MatmulImpl::OmpBlocked;
    if (std::strcmp(v, "omp_blocked_packb") == 0) return MatmulImpl::OmpBlockedPackB;
    if (std::strcmp(v, "omp_blocked_packab") == 0) return MatmulImpl::OmpBlockedPackAB;
    if (std::strcmp(v, "omp_gotoblas_avx2") == 0) return MatmulImpl::OmpGotoBlasAvx2;
    if (std::strcmp(v, "omp_gotoblas_avx512") == 0) return MatmulImpl::OmpGotoBlasAvx512;

    static bool warned = false;
    if (!warned) {
        warned = true;
        std::cerr << "[WARN] MATMUL_IMPL inconnu ('" << v
                  << "'). Utilisation de 'blas'. Valeurs valides : "
                  << "blas | ijk | ikj | blocked | omp | omp_blocked | "
                  << "omp_blocked_packb | omp_blocked_packab | "
                  << "omp_gotoblas_avx2 | omp_gotoblas_avx512\n";
    }
    return MatmulImpl::Blas;
}

MatmulImpl current_impl() {
    static const MatmulImpl impl = parse_impl_env();
    return impl;
}

size_t parse_block_size_env() {
    const char* v = std::getenv("MATMUL_BLOCK_SIZE");
    if (!v || !*v) {
        return static_cast<size_t>(BLOCK_SIZE);
    }

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(v, &end, 10);
    if (end == v || (end && *end != '\0') || parsed == 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[WARN] MATMUL_BLOCK_SIZE invalide ('" << v
                      << "'). Utilisation de la valeur par defaut "
                      << BLOCK_SIZE << ".\n";
        }
        return static_cast<size_t>(BLOCK_SIZE);
    }

    return static_cast<size_t>(parsed);
}

size_t current_block_size() {
    static const size_t bs = parse_block_size_env();
    return bs;
}

const char* matmul_impl_name(MatmulImpl impl) {
    switch (impl) {
        case MatmulImpl::Blas: return "blas";
        case MatmulImpl::Ijk: return "ijk";
        case MatmulImpl::Ikj: return "ikj";
        case MatmulImpl::Blocked: return "blocked";
        case MatmulImpl::Omp: return "omp";
        case MatmulImpl::OmpBlocked: return "omp_blocked";
        case MatmulImpl::OmpBlockedPackB: return "omp_blocked_packb";
        case MatmulImpl::OmpBlockedPackAB: return "omp_blocked_packab";
        case MatmulImpl::OmpGotoBlasAvx2: return "omp_gotoblas_avx2";
        case MatmulImpl::OmpGotoBlasAvx512: return "omp_gotoblas_avx512";
    }

    return "blas";
}

}  // namespace gemm

void Matrix::matmul_into(const Matrix& other, Matrix& out, bool transA, bool transB) const {
    const size_t M = transA ? cols : rows;
    const size_t K = transA ? rows : cols;
    const size_t Kb = transB ? other.cols : other.rows;
    const size_t N = transB ? other.rows : other.cols;

    if (K != Kb) {
        throw std::invalid_argument(
            "Multiplication matricielle impossible : dimensions incompatibles");
    }

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

    const gemm::MatmulImpl impl = gemm::current_impl();

    if (impl == gemm::MatmulImpl::Blas) {
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
            static_cast<int>(N));
    } else if (!transA && !transB) {
        switch (impl) {
            case gemm::MatmulImpl::Ijk:
                gemm::sgemm_ijk(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case gemm::MatmulImpl::Ikj:
                gemm::sgemm_ikj(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case gemm::MatmulImpl::Blocked:
                gemm::sgemm_blocked(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case gemm::MatmulImpl::Omp:
                gemm::sgemm_omp(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case gemm::MatmulImpl::OmpBlocked:
                gemm::sgemm_omp_blocked(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case gemm::MatmulImpl::OmpBlockedPackB:
                gemm::sgemm_omp_blocked_packb(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case gemm::MatmulImpl::OmpBlockedPackAB:
                gemm::sgemm_omp_blocked_packab(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case gemm::MatmulImpl::OmpGotoBlasAvx2:
                gemm::sgemm_omp_gotoblas_avx2(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case gemm::MatmulImpl::OmpGotoBlasAvx512:
                gemm::sgemm_omp_gotoblas_avx512(data.data(), other.data.data(), out.data.data(), M, N, K);
                break;
            case gemm::MatmulImpl::Blas:
                break;
        }
    } else {
        for (size_t i = 0; i < M; ++i) {
            Scalar* out_row = out.data.data() + i * N;
            for (size_t j = 0; j < N; ++j) {
                Scalar acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    const Scalar a = transA ? data[k * cols + i] : data[i * cols + k];
                    const Scalar b = transB ? other.data[j * other.cols + k]
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
    matmulProfileRecord(gemm::matmul_impl_name(impl), us);
#endif
}
