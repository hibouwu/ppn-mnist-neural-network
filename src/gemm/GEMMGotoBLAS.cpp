#include "gemm/matmul_internal.hpp"

#include <immintrin.h>
#include <omp.h>

#include <vector>

namespace gemm {

namespace {

constexpr size_t kAvx2Mr = 6;
constexpr size_t kAvx2Nr = 8;

void sgemm_tile_scalar(const Scalar* A,
                       const Scalar* packed_B,
                       Scalar* C,
                       size_t lda,
                       size_t ldc,
                       size_t Kc,
                       size_t rows,
                       size_t cols,
                       size_t packed_b_stride) {
    for (size_t i = 0; i < rows; ++i) {
        Scalar* Ci = C + i * ldc;
        const Scalar* Ai = A + i * lda;
        for (size_t j = 0; j < cols; ++j) {
            Scalar acc = Ci[j];
            for (size_t k = 0; k < Kc; ++k) {
                acc += Ai[k] * packed_B[k * packed_b_stride + j];
            }
            Ci[j] = acc;
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void microkernel_avx2_6x8(const Scalar* A,
                          const Scalar* packed_B,
                          Scalar* C,
                          size_t lda,
                          size_t ldc,
                          size_t Kc,
                          size_t packed_b_stride) {
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5 * ldc);

    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b = _mm256_loadu_ps(packed_B + k * packed_b_stride);

        const __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        const __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        const __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        const __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);
        const __m256 a4 = _mm256_set1_ps(A[4 * lda + k]);
        const __m256 a5 = _mm256_set1_ps(A[5 * lda + k]);

        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
        c4 = _mm256_fmadd_ps(a4, b, c4);
        c5 = _mm256_fmadd_ps(a5, b, c5);
    }

    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
    _mm256_storeu_ps(C + 4 * ldc, c4);
    _mm256_storeu_ps(C + 5 * ldc, c5);
}

}  // namespace

void sgemm_omp_gotoblas_avx2(const Scalar* A, const Scalar* B, Scalar* C,
                             size_t M, size_t N, size_t K) {
    const size_t MC = current_block_size();
    const size_t NC = current_block_size();
    const size_t KC = current_block_size();

    #pragma omp parallel
    {
        std::vector<Scalar> packed_B(KC * NC);
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const size_t num_i_blocks = (M + MC - 1) / MC;
        const size_t blocks_per_thread =
            (num_i_blocks + static_cast<size_t>(nthreads) - 1) / static_cast<size_t>(nthreads);

        const size_t block_begin = minz(static_cast<size_t>(tid) * blocks_per_thread, num_i_blocks);
        const size_t block_end = minz(block_begin + blocks_per_thread, num_i_blocks);

        for (size_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
            const size_t ii = block_idx * MC;
            const size_t i_max = minz(ii + MC, M);
            const size_t m_block = i_max - ii;

            for (size_t jj = 0; jj < N; jj += NC) {
                const size_t j_max = minz(jj + NC, N);
                const size_t n_block = j_max - jj;

                for (size_t i = ii; i < i_max; ++i) {
                    Scalar* Ci = C + i * N + jj;
                    for (size_t x = 0; x < n_block; ++x) {
                        Ci[x] = 0.0f;
                    }
                }

                for (size_t kk = 0; kk < K; kk += KC) {
                    const size_t k_max = minz(kk + KC, K);
                    const size_t k_block = k_max - kk;

                    for (size_t k_local = 0; k_local < k_block; ++k_local) {
                        const Scalar* Bsrc = B + (kk + k_local) * N + jj;
                        Scalar* Bdst = packed_B.data() + k_local * n_block;
                        for (size_t x = 0; x < n_block; ++x) {
                            Bdst[x] = Bsrc[x];
                        }
                    }

                    size_t ir = 0;
                    for (; ir + kAvx2Mr <= m_block; ir += kAvx2Mr) {
                        size_t jr = 0;
                        for (; jr + kAvx2Nr <= n_block; jr += kAvx2Nr) {
                            microkernel_avx2_6x8(
                                A + (ii + ir) * K + kk,
                                packed_B.data() + jr,
                                C + (ii + ir) * N + jj + jr,
                                K,
                                N,
                                k_block,
                                n_block);
                        }
                        if (jr < n_block) {
                            sgemm_tile_scalar(
                                A + (ii + ir) * K + kk,
                                packed_B.data() + jr,
                                C + (ii + ir) * N + jj + jr,
                                K,
                                N,
                                k_block,
                                kAvx2Mr,
                                n_block - jr,
                                n_block);
                        }
                    }

                    if (ir < m_block) {
                        sgemm_tile_scalar(
                            A + (ii + ir) * K + kk,
                            packed_B.data(),
                            C + (ii + ir) * N + jj,
                            K,
                            N,
                            k_block,
                            m_block - ir,
                            n_block,
                            n_block);
                    }
                }
            }
        }
    }
}

void sgemm_omp_gotoblas_avx512(const Scalar* A, const Scalar* B, Scalar* C,
                               size_t M, size_t N, size_t K) {
    sgemm_omp_blocked_packab(A, B, C, M, N, K);
}

}  // namespace gemm
