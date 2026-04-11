#include "gemm/matmul_internal.hpp"

#include <immintrin.h>
#include <omp.h>

#include <cstring>
#include <iostream>
#include <vector>

namespace gemm {

bool cpu_supports_avx2_fma() {
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
#else
    return false;
#endif
}

bool cpu_supports_avx512f() {
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
#else
    return false;
#endif
}

void pack_a_micro_panel(const Scalar* A,
                        Scalar* packed_A,
                        size_t lda,
                        size_t Kc,
                        size_t rows,
                        size_t mr) {
    for (size_t k = 0; k < Kc; ++k) {
        Scalar* dst = packed_A + k * mr;
        size_t i = 0;
        for (; i < rows; ++i) {
            dst[i] = A[i * lda + k];
        }
        for (; i < mr; ++i) {
            dst[i] = 0.0f;
        }
    }
}

void pack_b_micro_panel(const Scalar* B,
                        Scalar* packed_B,
                        size_t ldb,
                        size_t Kc,
                        size_t cols,
                        size_t nr) {
    for (size_t k = 0; k < Kc; ++k) {
        Scalar* dst = packed_B + k * nr;
        const Scalar* src = B + k * ldb;
        size_t j = 0;
        for (; j < cols; ++j) {
            dst[j] = src[j];
        }
        for (; j < nr; ++j) {
            dst[j] = 0.0f;
        }
    }
}

void sgemm_tile_scalar_packed(const Scalar* packed_A,
                              const Scalar* packed_B,
                              Scalar* C,
                              size_t ldc,
                              size_t Kc,
                              size_t rows,
                              size_t cols,
                              size_t mr,
                              size_t nr) {
    for (size_t i = 0; i < rows; ++i) {
        Scalar* Ci = C + i * ldc;
        for (size_t j = 0; j < cols; ++j) {
            Scalar acc = Ci[j];
            for (size_t k = 0; k < Kc; ++k) {
                acc += packed_A[k * mr + i] * packed_B[k * nr + j];
            }
            Ci[j] = acc;
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void microkernel_avx2_6x8(const Scalar* packed_A,
                          const Scalar* packed_B,
                          Scalar* C,
                          size_t ldc,
                          size_t Kc) {
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5 * ldc);

    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b = _mm256_loadu_ps(packed_B + k * 8);
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 6 + 0]), b, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 6 + 1]), b, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 6 + 2]), b, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 6 + 3]), b, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 6 + 4]), b, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 6 + 5]), b, c5);
    }

    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
    _mm256_storeu_ps(C + 4 * ldc, c4);
    _mm256_storeu_ps(C + 5 * ldc, c5);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void microkernel_avx2_8x8(const Scalar* packed_A,
                          const Scalar* packed_B,
                          Scalar* C,
                          size_t ldc,
                          size_t Kc) {
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 c6 = _mm256_loadu_ps(C + 6 * ldc);
    __m256 c7 = _mm256_loadu_ps(C + 7 * ldc);

    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b = _mm256_loadu_ps(packed_B + k * 8);
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 8 + 0]), b, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 8 + 1]), b, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 8 + 2]), b, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 8 + 3]), b, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 8 + 4]), b, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 8 + 5]), b, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 8 + 6]), b, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 8 + 7]), b, c7);
    }

    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
    _mm256_storeu_ps(C + 4 * ldc, c4);
    _mm256_storeu_ps(C + 5 * ldc, c5);
    _mm256_storeu_ps(C + 6 * ldc, c6);
    _mm256_storeu_ps(C + 7 * ldc, c7);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void microkernel_avx2_4x16(const Scalar* packed_A,
                           const Scalar* packed_B,
                           Scalar* C,
                           size_t ldc,
                           size_t Kc) {
    __m256 c00 = _mm256_loadu_ps(C + 0 * ldc + 0);
    __m256 c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
    __m256 c10 = _mm256_loadu_ps(C + 1 * ldc + 0);
    __m256 c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2 * ldc + 0);
    __m256 c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3 * ldc + 0);
    __m256 c31 = _mm256_loadu_ps(C + 3 * ldc + 8);

    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(packed_B + k * 16 + 0);
        const __m256 b1 = _mm256_loadu_ps(packed_B + k * 16 + 8);
        c00 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 4 + 0]), b0, c00);
        c01 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 4 + 0]), b1, c01);
        c10 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 4 + 1]), b0, c10);
        c11 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 4 + 1]), b1, c11);
        c20 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 4 + 2]), b0, c20);
        c21 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 4 + 2]), b1, c21);
        c30 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 4 + 3]), b0, c30);
        c31 = _mm256_fmadd_ps(_mm256_set1_ps(packed_A[k * 4 + 3]), b1, c31);
    }

    _mm256_storeu_ps(C + 0 * ldc + 0, c00);
    _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc + 0, c10);
    _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc + 0, c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc + 0, c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, c31);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif
void microkernel_avx512_8x16(const Scalar* packed_A,
                             const Scalar* packed_B,
                             Scalar* C,
                             size_t ldc,
                             size_t Kc) {
    __m512 c0 = _mm512_loadu_ps(C + 0 * ldc);
    __m512 c1 = _mm512_loadu_ps(C + 1 * ldc);
    __m512 c2 = _mm512_loadu_ps(C + 2 * ldc);
    __m512 c3 = _mm512_loadu_ps(C + 3 * ldc);
    __m512 c4 = _mm512_loadu_ps(C + 4 * ldc);
    __m512 c5 = _mm512_loadu_ps(C + 5 * ldc);
    __m512 c6 = _mm512_loadu_ps(C + 6 * ldc);
    __m512 c7 = _mm512_loadu_ps(C + 7 * ldc);

    for (size_t k = 0; k < Kc; ++k) {
        const __m512 b = _mm512_loadu_ps(packed_B + k * 16);
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(packed_A[k * 8 + 0]), b, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(packed_A[k * 8 + 1]), b, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(packed_A[k * 8 + 2]), b, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(packed_A[k * 8 + 3]), b, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(packed_A[k * 8 + 4]), b, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(packed_A[k * 8 + 5]), b, c5);
        c6 = _mm512_fmadd_ps(_mm512_set1_ps(packed_A[k * 8 + 6]), b, c6);
        c7 = _mm512_fmadd_ps(_mm512_set1_ps(packed_A[k * 8 + 7]), b, c7);
    }

    _mm512_storeu_ps(C + 0 * ldc, c0);
    _mm512_storeu_ps(C + 1 * ldc, c1);
    _mm512_storeu_ps(C + 2 * ldc, c2);
    _mm512_storeu_ps(C + 3 * ldc, c3);
    _mm512_storeu_ps(C + 4 * ldc, c4);
    _mm512_storeu_ps(C + 5 * ldc, c5);
    _mm512_storeu_ps(C + 6 * ldc, c6);
    _mm512_storeu_ps(C + 7 * ldc, c7);
}

constexpr KernelSpec kKernelAvx2_6x8{"avx2_6x8", 6, 8, KernelIsa::Avx2, microkernel_avx2_6x8};
constexpr KernelSpec kKernelAvx2_8x8{"avx2_8x8", 8, 8, KernelIsa::Avx2, microkernel_avx2_8x8};
constexpr KernelSpec kKernelAvx2_4x16{"avx2_4x16", 4, 16, KernelIsa::Avx2, microkernel_avx2_4x16};
constexpr KernelSpec kKernelAvx512_8x16{"avx512_8x16", 8, 16, KernelIsa::Avx512, microkernel_avx512_8x16};

const KernelSpec* find_kernel_by_name(const char* name) {
    static constexpr const KernelSpec* kAll[] = {
        &kKernelAvx2_6x8,
        &kKernelAvx2_8x8,
        &kKernelAvx2_4x16,
        &kKernelAvx512_8x16,
    };

    for (const KernelSpec* spec : kAll) {
        if (std::strcmp(spec->name, name) == 0) {
            return spec;
        }
    }
    return nullptr;
}

const KernelSpec& default_kernel_for_isa(KernelIsa isa) {
    return isa == KernelIsa::Avx512 ? kKernelAvx512_8x16 : kKernelAvx2_6x8;
}

const KernelSpec& current_kernel_for_isa(KernelIsa isa) {
    const char* v = std::getenv("MATMUL_GOTO_KERNEL");
    if (!v || !*v) {
        return default_kernel_for_isa(isa);
    }

    const KernelSpec* selected = find_kernel_by_name(v);
    if (!selected || selected->isa != isa) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[WARN] MATMUL_GOTO_KERNEL incompatible ('" << v
                      << "'). Using default kernel for selected ISA.\n";
        }
        return default_kernel_for_isa(isa);
    }

    return *selected;
}

namespace {

void run_gotoblas_kernel(const KernelSpec& kernel,
                         const Scalar* A,
                         const Scalar* B,
                         Scalar* C,
                         size_t M,
                         size_t N,
                         size_t K) {
    const size_t MC = current_pack_m_block_size();
    const size_t NC = current_pack_n_block_size();
    const size_t KC = current_pack_k_block_size();

    #pragma omp parallel
    {
        const size_t max_m_panels = (MC + kernel.mr - 1) / kernel.mr;
        const size_t max_n_panels = (NC + kernel.nr - 1) / kernel.nr;
        std::vector<Scalar> packed_A(max_m_panels * KC * kernel.mr);
        std::vector<Scalar> packed_B(max_n_panels * KC * kernel.nr);
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
                    const size_t m_panels = (m_block + kernel.mr - 1) / kernel.mr;
                    const size_t n_panels = (n_block + kernel.nr - 1) / kernel.nr;

                    for (size_t jr_panel = 0; jr_panel < n_panels; ++jr_panel) {
                        const size_t jr = jr_panel * kernel.nr;
                        const size_t cols = minz(kernel.nr, n_block - jr);
                        pack_b_micro_panel(
                            B + kk * N + jj + jr,
                            packed_B.data() + jr_panel * k_block * kernel.nr,
                            N,
                            k_block,
                            cols,
                            kernel.nr);
                    }

                    for (size_t ir_panel = 0; ir_panel < m_panels; ++ir_panel) {
                        const size_t ir = ir_panel * kernel.mr;
                        const size_t rows = minz(kernel.mr, m_block - ir);
                        pack_a_micro_panel(
                            A + (ii + ir) * K + kk,
                            packed_A.data() + ir_panel * k_block * kernel.mr,
                            K,
                            k_block,
                            rows,
                            kernel.mr);

                        const Scalar* packed_a_panel =
                            packed_A.data() + ir_panel * k_block * kernel.mr;

                        for (size_t jr_panel = 0; jr_panel < n_panels; ++jr_panel) {
                            const size_t jr = jr_panel * kernel.nr;
                            const size_t cols = minz(kernel.nr, n_block - jr);
                            const Scalar* packed_b_panel =
                                packed_B.data() + jr_panel * k_block * kernel.nr;
                            Scalar* Ctile = C + (ii + ir) * N + jj + jr;

                            if (rows == kernel.mr && cols == kernel.nr) {
                                kernel.fn(packed_a_panel, packed_b_panel, Ctile, N, k_block);
                            } else {
                                sgemm_tile_scalar_packed(
                                    packed_a_panel,
                                    packed_b_panel,
                                    Ctile,
                                    N,
                                    k_block,
                                    rows,
                                    cols,
                                    kernel.mr,
                                    kernel.nr);
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

void sgemm_omp_gotoblas_avx2(const Scalar* A, const Scalar* B, Scalar* C,
                             size_t M, size_t N, size_t K) {
    if (!cpu_supports_avx2_fma()) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[WARN] CPU lacks AVX2/FMA support. Falling back to omp_blocked_packab.\n";
        }
        sgemm_omp_blocked_packab(A, B, C, M, N, K);
        return;
    }

    run_gotoblas_kernel(current_kernel_for_isa(KernelIsa::Avx2), A, B, C, M, N, K);
}

void sgemm_omp_gotoblas_avx512(const Scalar* A, const Scalar* B, Scalar* C,
                               size_t M, size_t N, size_t K) {
    if (!cpu_supports_avx512f()) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[WARN] CPU lacks AVX-512F support. Falling back to omp_gotoblas_avx2.\n";
        }
        sgemm_omp_gotoblas_avx2(A, B, C, M, N, K);
        return;
    }

    run_gotoblas_kernel(current_kernel_for_isa(KernelIsa::Avx512), A, B, C, M, N, K);
}

}  // namespace gemm
