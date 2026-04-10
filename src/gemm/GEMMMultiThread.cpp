#include "gemm/matmul_internal.hpp"

#include <omp.h>

#include <vector>

namespace gemm {

void sgemm_omp(const Scalar* A, const Scalar* B, Scalar* C,
               size_t M, size_t N, size_t K) {
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        const size_t rows_per_thread =
            (M + static_cast<size_t>(nthreads) - 1) / static_cast<size_t>(nthreads);
        const size_t row_begin = minz(static_cast<size_t>(tid) * rows_per_thread, M);
        const size_t row_end = minz(row_begin + rows_per_thread, M);

        for (size_t i = row_begin; i < row_end; ++i) {
            Scalar* Ci = C + i * N;
            for (size_t j = 0; j < N; ++j) {
                Ci[j] = 0.0f;
            }
        }

        for (size_t i = row_begin; i < row_end; ++i) {
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
}

void sgemm_omp_blocked(const Scalar* A, const Scalar* B, Scalar* C,
                       size_t M, size_t N, size_t K) {
    const size_t BS = current_block_size();

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const size_t num_i_blocks = (M + BS - 1) / BS;
        const size_t blocks_per_thread =
            (num_i_blocks + static_cast<size_t>(nthreads) - 1) / static_cast<size_t>(nthreads);

        const size_t block_begin = minz(static_cast<size_t>(tid) * blocks_per_thread, num_i_blocks);
        const size_t block_end = minz(block_begin + blocks_per_thread, num_i_blocks);

        for (size_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
            const size_t ii = block_idx * BS;
            const size_t i_max = minz(ii + BS, M);

            for (size_t jj = 0; jj < N; jj += BS) {
                const size_t j_max = minz(jj + BS, N);
                const size_t len = j_max - jj;

                for (size_t i = ii; i < i_max; ++i) {
                    Scalar* Ci = C + i * N + jj;
                    for (size_t x = 0; x < len; ++x) {
                        Ci[x] = 0.0f;
                    }
                }

                for (size_t kk = 0; kk < K; kk += BS) {
                    const size_t k_max = minz(kk + BS, K);

                    for (size_t i = ii; i < i_max; ++i) {
                        Scalar* Ci = C + i * N + jj;
                        for (size_t k = kk; k < k_max; ++k) {
                            const Scalar aik = A[i * K + k];
                            const Scalar* Bk = B + k * N + jj;

                            for (size_t x = 0; x < len; ++x) {
                                Ci[x] += aik * Bk[x];
                            }
                        }
                    }
                }
            }
        }
    }
}

void sgemm_omp_blocked_packb(const Scalar* A, const Scalar* B, Scalar* C,
                             size_t M, size_t N, size_t K) {
    const size_t BS = current_block_size();

    #pragma omp parallel
    {
        std::vector<Scalar> packed_B(BS * BS);
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const size_t num_i_blocks = (M + BS - 1) / BS;
        const size_t blocks_per_thread =
            (num_i_blocks + static_cast<size_t>(nthreads) - 1) / static_cast<size_t>(nthreads);

        const size_t block_begin = minz(static_cast<size_t>(tid) * blocks_per_thread, num_i_blocks);
        const size_t block_end = minz(block_begin + blocks_per_thread, num_i_blocks);

        for (size_t block_idx = block_begin; block_idx < block_end; ++block_idx) {
            const size_t ii = block_idx * BS;
            const size_t i_max = minz(ii + BS, M);

            for (size_t jj = 0; jj < N; jj += BS) {
                const size_t j_max = minz(jj + BS, N);
                const size_t n_block = j_max - jj;

                for (size_t i = ii; i < i_max; ++i) {
                    Scalar* Ci = C + i * N + jj;
                    for (size_t x = 0; x < n_block; ++x) {
                        Ci[x] = 0.0f;
                    }
                }

                for (size_t kk = 0; kk < K; kk += BS) {
                    const size_t k_max = minz(kk + BS, K);
                    const size_t k_block = k_max - kk;

                    for (size_t k_local = 0; k_local < k_block; ++k_local) {
                        const Scalar* Bsrc = B + (kk + k_local) * N + jj;
                        Scalar* Bdst = packed_B.data() + k_local * n_block;
                        for (size_t x = 0; x < n_block; ++x) {
                            Bdst[x] = Bsrc[x];
                        }
                    }

                    for (size_t i = ii; i < i_max; ++i) {
                        const Scalar* Ai = A + i * K + kk;
                        Scalar* Ci = C + i * N + jj;

                        for (size_t k_local = 0; k_local < k_block; ++k_local) {
                            const Scalar aik = Ai[k_local];
                            const Scalar* Bp = packed_B.data() + k_local * n_block;

                            for (size_t x = 0; x < n_block; ++x) {
                                Ci[x] += aik * Bp[x];
                            }
                        }
                    }
                }
            }
        }
    }
}

void sgemm_omp_blocked_packab(const Scalar* A, const Scalar* B, Scalar* C,
                              size_t M, size_t N, size_t K) {
    sgemm_omp_blocked_packb(A, B, C, M, N, K);
}

}  // namespace gemm
