#include "gemm/matmul_internal.hpp"

namespace gemm {

void sgemm_ijk(const Scalar* A, const Scalar* B, Scalar* C,
               size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            Scalar acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

void sgemm_ikj(
#if defined(__GNUC__) || defined(__clang__)
    const Scalar* __restrict A,
    const Scalar* __restrict B,
#else
    const Scalar* A,
    const Scalar* B,
#endif
    Scalar* C,
    size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        Scalar* Ci = C + i * N;
        for (size_t j = 0; j < N; ++j) {
            Ci[j] = 0.0f;
        }
    }

    for (size_t i = 0; i < M; ++i) {
        const Scalar* Ai = A + i * K;
        Scalar* Ci = C + i * N;
        for (size_t k = 0; k < K; ++k) {
            const Scalar aik = Ai[k];
            const Scalar* Bk = B + k * N;
            for (size_t j = 0; j < N; ++j) {
                Ci[j] += aik * Bk[j];
            }
        }
    }
}

void sgemm_blocked(const Scalar* A, const Scalar* B, Scalar* C,
                   size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        Scalar* Ci = C + i * N;
        for (size_t j = 0; j < N; ++j) {
            Ci[j] = 0.0f;
        }
    }

    const size_t BS = current_block_size();

    for (size_t ii = 0; ii < M; ii += BS) {
        const size_t i_max = minz(ii + BS, M);
        for (size_t kk = 0; kk < K; kk += BS) {
            const size_t k_max = minz(kk + BS, K);
            for (size_t jj = 0; jj < N; jj += BS) {
                const size_t j_max = minz(jj + BS, N);
                const size_t len = j_max - jj;

                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        const Scalar aik = A[i * K + k];
                        const Scalar* Bk = B + k * N + jj;
                        Scalar* Ci = C + i * N + jj;

                        for (size_t x = 0; x < len; ++x) {
                            Ci[x] += aik * Bk[x];
                        }
                    }
                }
            }
        }
    }
}

}  // namespace gemm
