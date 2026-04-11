#ifndef GEMM_MATMUL_INTERNAL_HPP
#define GEMM_MATMUL_INTERNAL_HPP

#include "tensor.hpp"

#include <cstddef>

namespace gemm {

using MicroKernelFn = void (*)(const Scalar*, const Scalar*, Scalar*, size_t, size_t);

enum class KernelIsa {
    Avx2,
    Avx512,
};

struct KernelSpec {
    const char* name;
    size_t mr;
    size_t nr;
    KernelIsa isa;
    MicroKernelFn fn;
};

enum class MatmulImpl {
    Blas,
    Ijk,
    Ikj,
    Blocked,
    Omp,
    OmpBlocked,
    OmpBlockedPackB,
    OmpBlockedPackAB,
    OmpGotoBlasAvx2,
    OmpGotoBlasAvx512,
};

size_t minz(size_t a, size_t b);
MatmulImpl parse_impl_env();
MatmulImpl current_impl();
size_t parse_block_size_env();
size_t current_block_size();
size_t current_pack_m_block_size();
size_t current_pack_n_block_size();
size_t current_pack_k_block_size();
const char* matmul_impl_name(MatmulImpl impl);

void sgemm_ijk(const Scalar* A, const Scalar* B, Scalar* C,
               size_t M, size_t N, size_t K);
void sgemm_ikj(const Scalar* A, const Scalar* B, Scalar* C,
               size_t M, size_t N, size_t K);
void sgemm_blocked(const Scalar* A, const Scalar* B, Scalar* C,
                   size_t M, size_t N, size_t K);

void sgemm_omp(const Scalar* A, const Scalar* B, Scalar* C,
               size_t M, size_t N, size_t K);
void sgemm_omp_blocked(const Scalar* A, const Scalar* B, Scalar* C,
                       size_t M, size_t N, size_t K);
void sgemm_omp_blocked_packb(const Scalar* A, const Scalar* B, Scalar* C,
                             size_t M, size_t N, size_t K);
void sgemm_omp_blocked_packab(const Scalar* A, const Scalar* B, Scalar* C,
                              size_t M, size_t N, size_t K);

void sgemm_omp_gotoblas_avx2(const Scalar* A, const Scalar* B, Scalar* C,
                             size_t M, size_t N, size_t K);
void sgemm_omp_gotoblas_avx512(const Scalar* A, const Scalar* B, Scalar* C,
                               size_t M, size_t N, size_t K);

bool cpu_supports_avx2_fma();
bool cpu_supports_avx512f();

void pack_a_micro_panel(const Scalar* A, Scalar* packed_A,
                        size_t lda, size_t Kc, size_t rows, size_t mr);
void pack_b_micro_panel(const Scalar* B, Scalar* packed_B,
                        size_t ldb, size_t Kc, size_t cols, size_t nr);

void sgemm_tile_scalar_packed(const Scalar* packed_A, const Scalar* packed_B,
                              Scalar* C, size_t ldc, size_t Kc,
                              size_t rows, size_t cols,
                              size_t mr, size_t nr);

void microkernel_avx2_6x8(const Scalar* packed_A, const Scalar* packed_B,
                          Scalar* C, size_t ldc, size_t Kc);
void microkernel_avx2_8x8(const Scalar* packed_A, const Scalar* packed_B,
                          Scalar* C, size_t ldc, size_t Kc);
void microkernel_avx2_4x16(const Scalar* packed_A, const Scalar* packed_B,
                           Scalar* C, size_t ldc, size_t Kc);
void microkernel_avx512_8x16(const Scalar* packed_A, const Scalar* packed_B,
                             Scalar* C, size_t ldc, size_t Kc);

const KernelSpec* find_kernel_by_name(const char* name);
const KernelSpec& default_kernel_for_isa(KernelIsa isa);
const KernelSpec& current_kernel_for_isa(KernelIsa isa);

}  // namespace gemm

#endif
