#ifndef GEMM_MATMUL_INTERNAL_HPP
#define GEMM_MATMUL_INTERNAL_HPP

#include "tensor.hpp"

#include <cstddef>

namespace gemm {

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
size_t current_mc_block_size();
size_t current_nc_block_size();
size_t current_kc_block_size();
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
size_t gotoblas_default_mr();
size_t gotoblas_default_nr();
size_t gotoblas_default_mc();
size_t gotoblas_default_nc();
size_t gotoblas_default_kc();
const char* gotoblas_current_kernel_name();
size_t gotoblas_current_mr();
size_t gotoblas_current_nr();
bool gotoblas_try_get_kernel_shape(const char* shape_name, size_t* mr, size_t* nr);

void pack_a_micro_panel(const Scalar* A, Scalar* packed_A,
                        size_t lda, size_t Kc, size_t rows, size_t mr);
void pack_b_micro_panel(const Scalar* B, Scalar* packed_B,
                        size_t ldb, size_t Kc, size_t cols, size_t nr);

void gotoblas_microkernel_fixed(const Scalar* packed_A,
                                const Scalar* packed_B,
                                Scalar* C,
                                size_t ldc,
                                size_t Kc,
                                size_t rows,
                                size_t cols);
void gotoblas_microkernel_for_shape(const char* shape_name,
                                    const Scalar* packed_A,
                                    const Scalar* packed_B,
                                    Scalar* C,
                                    size_t ldc,
                                    size_t Kc,
                                    size_t rows,
                                    size_t cols);

}  // namespace gemm

#endif
