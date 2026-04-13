#include "gemm/matmul_internal.hpp"

#include <immintrin.h>
#include <omp.h>

#include <atomic>
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
std::atomic<size_t> g_gotoblas_pack_bc_call_count{0};

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

void reset_gotoblas_debug_counters() {
    g_gotoblas_pack_bc_call_count.store(0, std::memory_order_relaxed);
}

size_t gotoblas_pack_bc_call_count() {
    return g_gotoblas_pack_bc_call_count.load(std::memory_order_relaxed);
}

namespace {

void run_micro_kernel(const KernelSpec& kernel,
                      const Scalar* packed_Ac_panel,
                      const Scalar* packed_Bc_panel,
                      Scalar* Ctile,
                      size_t ldc,
                      size_t Kc,
                      size_t rows,
                      size_t cols) {
    if (rows == kernel.mr && cols == kernel.nr) {
        kernel.fn(packed_Ac_panel, packed_Bc_panel, Ctile, ldc, Kc);
        return;
    }

    sgemm_tile_scalar_packed(
        packed_Ac_panel,
        packed_Bc_panel,
        Ctile,
        ldc,
        Kc,
        rows,
        cols,
        kernel.mr,
        kernel.nr);
}

void pack_Bc(const KernelSpec& kernel,
             const Scalar* B_block,
             Scalar* packed_Bc,
             size_t ldb,
             size_t Kc,
             size_t Nc) {
    const size_t nr_panels = (Nc + kernel.nr - 1) / kernel.nr;

    for (size_t jr = 0; jr < nr_panels; ++jr) {
        const size_t jc_inner = jr * kernel.nr;
        const size_t cols = minz(kernel.nr, Nc - jc_inner);
        pack_b_micro_panel(
            B_block + jc_inner,
            packed_Bc + jr * Kc * kernel.nr,
            ldb,
            Kc,
            cols,
            kernel.nr);
    }
}

void pack_Ac(const KernelSpec& kernel,
             const Scalar* A_block,
             Scalar* packed_Ac,
             size_t lda,
             size_t Mc,
             size_t Kc) {
    const size_t mr_panels = (Mc + kernel.mr - 1) / kernel.mr;

    for (size_t ir = 0; ir < mr_panels; ++ir) {
        const size_t ic_inner = ir * kernel.mr;
        const size_t rows = minz(kernel.mr, Mc - ic_inner);
        pack_a_micro_panel(
            A_block + ic_inner * lda,
            packed_Ac + ir * Kc * kernel.mr,
            lda,
            Kc,
            rows,
            kernel.mr);
    }
}

void run_macro_kernel(const KernelSpec& kernel,
                      const Scalar* packed_Ac,
                      const Scalar* packed_Bc,
                      Scalar* C_block,
                      size_t ldc,
                      size_t Mc,
                      size_t Nc,
                      size_t Kc) {
    const size_t mr_panels = (Mc + kernel.mr - 1) / kernel.mr;
    const size_t nr_panels = (Nc + kernel.nr - 1) / kernel.nr;

    for (size_t jr = 0; jr < nr_panels; ++jr) {
        const size_t jc_inner = jr * kernel.nr;
        const size_t cols = minz(kernel.nr, Nc - jc_inner);
        const Scalar* packed_Bc_panel = packed_Bc + jr * Kc * kernel.nr;

        for (size_t ir = 0; ir < mr_panels; ++ir) {
            const size_t ic_inner = ir * kernel.mr;
            const size_t rows = minz(kernel.mr, Mc - ic_inner);
            const Scalar* packed_Ac_panel = packed_Ac + ir * Kc * kernel.mr;
            Scalar* Ctile = C_block + ic_inner * ldc + jc_inner;

            run_micro_kernel(
                kernel,
                packed_Ac_panel,
                packed_Bc_panel,
                Ctile,
                ldc,
                Kc,
                rows,
                cols);
        }
    }
}

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
    const size_t max_mr_panels = (MC + kernel.mr - 1) / kernel.mr;
    const size_t max_nr_panels = (NC + kernel.nr - 1) / kernel.nr;
    const size_t num_ic_blocks = (M + MC - 1) / MC;

    for (size_t jc = 0; jc < N; jc += NC) {
        const size_t j_end = minz(jc + NC, N);
        const size_t nc_cur = j_end - jc;
        std::vector<Scalar> packed_Bc(max_nr_panels * KC * kernel.nr);

        #pragma omp parallel
        {
            std::vector<Scalar> packed_Ac(max_mr_panels * KC * kernel.mr);

            for (size_t pc = 0; pc < K; pc += KC) {
                const size_t p_end = minz(pc + KC, K);
                const size_t kc_cur = p_end - pc;

                // Keep the implicit barrier on `single`: the following `omp for`
                // reads the shared packed_Bc buffer and must not observe a partial pack.
                // Do not add `nowait` here without replacing the synchronization.
                #pragma omp single
                {
                    pack_Bc(kernel, B + pc * N + jc, packed_Bc.data(), N, kc_cur, nc_cur);
                    g_gotoblas_pack_bc_call_count.fetch_add(1, std::memory_order_relaxed);
                }

                // Keep the implicit barrier on `omp for`: all ic-block consumers of the
                // current packed_Bc must finish before the next pc iteration repacks it.
                // Do not weaken this synchronization casually.
                #pragma omp for schedule(static)
                for (size_t ic_block = 0; ic_block < num_ic_blocks; ++ic_block) {
                    const size_t ic = ic_block * MC;
                    const size_t i_end = minz(ic + MC, M);
                    const size_t mc_cur = i_end - ic;

                    if (pc == 0) {
                        for (size_t i = ic; i < i_end; ++i) {
                            Scalar* Ci = C + i * N + jc;
                            for (size_t x = 0; x < nc_cur; ++x) {
                                Ci[x] = 0.0f;
                            }
                        }
                    }

                    pack_Ac(kernel, A + ic * K + pc, packed_Ac.data(), K, mc_cur, kc_cur);
                    run_macro_kernel(
                        kernel,
                        packed_Ac.data(),
                        packed_Bc.data(),
                        C + ic * N + jc,
                        N,
                        mc_cur,
                        nc_cur,
                        kc_cur);
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
