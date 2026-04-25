#include "gemm/matmul_internal.hpp"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>

namespace gemm {

namespace {

constexpr size_t kDefaultMicroKernelRows = 8; // Mr = 8
constexpr size_t kDefaultMicroKernelCols = 8; // Nr = 8
constexpr size_t kCacheBlockM = 8; // Mc = 64
constexpr size_t kCacheBlockN = 448; // Nc = 448
constexpr size_t kCacheBlockK = 384; // Kc = 384
constexpr size_t kPackedWorkspaceAlignment = 64;

enum class KernelIsa {
    Avx2,
    Avx512,
};

struct KernelShape {
    const char* name;
    KernelIsa isa;
    size_t mr;
    size_t nr;
};

constexpr KernelShape kKernel8x8{"avx2_8x8", KernelIsa::Avx2, 8, 8};
constexpr KernelShape kKernel12x8{"avx2_12x8", KernelIsa::Avx2, 12, 8};
constexpr KernelShape kKernel13x8{"avx2_13x8", KernelIsa::Avx2, 13, 8};
constexpr KernelShape kKernel4x16{"avx2_4x16", KernelIsa::Avx2, 4, 16};
constexpr KernelShape kKernel5x16{"avx2_5x16", KernelIsa::Avx2, 5, 16};
constexpr KernelShape kKernel6x16{"avx2_6x16", KernelIsa::Avx2, 6, 16};
constexpr KernelShape kKernelAvx5124x16{"avx512_4x16", KernelIsa::Avx512, 4, 16};
constexpr KernelShape kKernelAvx5128x16{"avx512_8x16", KernelIsa::Avx512, 8, 16};
constexpr KernelShape kKernelAvx51214x16{"avx512_14x16", KernelIsa::Avx512, 14, 16};
constexpr KernelShape kKernelAvx51216x16{"avx512_16x16", KernelIsa::Avx512, 16, 16};
constexpr KernelShape kKernelAvx51218x16{"avx512_18x16", KernelIsa::Avx512, 18, 16};
constexpr KernelShape kKernelAvx51220x16{"avx512_20x16", KernelIsa::Avx512, 20, 16};
constexpr KernelShape kKernelAvx5124x32{"avx512_4x32", KernelIsa::Avx512, 4, 32};
constexpr KernelShape kKernelAvx5126x32{"avx512_6x32", KernelIsa::Avx512, 6, 32};
constexpr KernelShape kKernelAvx5128x32{"avx512_8x32", KernelIsa::Avx512, 8, 32};
constexpr KernelShape kKernelAvx51210x32{"avx512_10x32", KernelIsa::Avx512, 10, 32};
constexpr KernelShape kKernelAvx51212x32{"avx512_12x32", KernelIsa::Avx512, 12, 32};

const KernelShape* find_kernel_shape(const char* shape_name) {
    if (shape_name == nullptr || *shape_name == '\0' ||
        std::strcmp(shape_name, "avx2_8x8") == 0 ||
        std::strcmp(shape_name, "8x8") == 0) {
        return &kKernel8x8;
    }
    if (std::strcmp(shape_name, "avx2_4x16") == 0 ||
        std::strcmp(shape_name, "4x16") == 0) {
        return &kKernel4x16;
    }
    if (std::strcmp(shape_name, "avx2_12x8") == 0 ||
        std::strcmp(shape_name, "12x8") == 0) {
        return &kKernel12x8;
    }
    if (std::strcmp(shape_name, "avx2_13x8") == 0 ||
        std::strcmp(shape_name, "13x8") == 0) {
        return &kKernel13x8;
    }
    if (std::strcmp(shape_name, "avx2_5x16") == 0 ||
        std::strcmp(shape_name, "5x16") == 0) {
        return &kKernel5x16;
    }
    if (std::strcmp(shape_name, "avx2_6x16") == 0 ||
        std::strcmp(shape_name, "6x16") == 0) {
        return &kKernel6x16;
    }
    if (std::strcmp(shape_name, "avx512_8x16") == 0) {
        return &kKernelAvx5128x16;
    }
    if (std::strcmp(shape_name, "avx512_4x16") == 0) {
        return &kKernelAvx5124x16;
    }
    if (std::strcmp(shape_name, "avx512_14x16") == 0) {
        return &kKernelAvx51214x16;
    }
    if (std::strcmp(shape_name, "avx512_16x16") == 0) {
        return &kKernelAvx51216x16;
    }
    if (std::strcmp(shape_name, "avx512_18x16") == 0) {
        return &kKernelAvx51218x16;
    }
    if (std::strcmp(shape_name, "avx512_20x16") == 0) {
        return &kKernelAvx51220x16;
    }
    if (std::strcmp(shape_name, "avx512_4x32") == 0) {
        return &kKernelAvx5124x32;
    }
    if (std::strcmp(shape_name, "avx512_6x32") == 0) {
        return &kKernelAvx5126x32;
    }
    if (std::strcmp(shape_name, "avx512_8x32") == 0) {
        return &kKernelAvx5128x32;
    }
    if (std::strcmp(shape_name, "avx512_10x32") == 0) {
        return &kKernelAvx51210x32;
    }
    if (std::strcmp(shape_name, "avx512_12x32") == 0) {
        return &kKernelAvx51212x32;
    }
    return nullptr;
}

const char* kernel_isa_name(KernelIsa isa) {
    return isa == KernelIsa::Avx512 ? "AVX-512" : "AVX2";
}

const KernelShape& current_kernel_shape(KernelIsa isa) {
    const char* env = std::getenv("MATMUL_GOTO_KERNEL");
    const KernelShape* resolved = find_kernel_shape(env);
    if (resolved != nullptr) {
        if (resolved->isa != isa) {
            throw std::invalid_argument(
                std::string("MATMUL_GOTO_KERNEL '") + resolved->name +
                "' is not valid for the " + kernel_isa_name(isa) + " GotoBLAS path");
        }
        return *resolved;
    }
    if (isa == KernelIsa::Avx512) {
        if (env != nullptr && *env != '\0') {
            throw std::invalid_argument(
                std::string("Unknown AVX-512 GotoBLAS micro-kernel shape: '") + env + "'");
        }
        return kKernelAvx5128x32;
    }
    static bool warned = false;
    if (!warned && env != nullptr && *env != '\0') {
        warned = true;
        std::cerr << "[WARN] MATMUL_GOTO_KERNEL inconnu ('" << env
                  << "'). Utilisation de avx2_8x8.\n";
    }
    return kKernel8x8;
}

const KernelShape& current_kernel_shape() {
    return current_kernel_shape(KernelIsa::Avx2);
}

struct PackedWorkspace {
    Scalar* packed_A = nullptr;
    Scalar* packed_B = nullptr;
    size_t packed_A_capacity = 0;
    size_t packed_B_capacity = 0;

    ~PackedWorkspace() {
        if (packed_A) {
            ::operator delete[](packed_A, std::align_val_t(kPackedWorkspaceAlignment));
        }
        if (packed_B) {
            ::operator delete[](packed_B, std::align_val_t(kPackedWorkspaceAlignment));
        }
    }

    Scalar* ensure_packed_a(size_t elements) {
        if (elements > packed_A_capacity) {
            if (packed_A) {
                ::operator delete[](packed_A, std::align_val_t(kPackedWorkspaceAlignment));
            }
            packed_A = static_cast<Scalar*>(
                ::operator new[](elements * sizeof(Scalar),
                                  std::align_val_t(kPackedWorkspaceAlignment)));
            packed_A_capacity = elements;
        }
        return packed_A;
    }

    Scalar* ensure_packed_b(size_t elements) {
        if (elements > packed_B_capacity) {
            if (packed_B) {
                ::operator delete[](packed_B, std::align_val_t(kPackedWorkspaceAlignment));
            }
            packed_B = static_cast<Scalar*>(
                ::operator new[](elements * sizeof(Scalar),
                                  std::align_val_t(kPackedWorkspaceAlignment)));
            packed_B_capacity = elements;
        }
        return packed_B;
    }
};

PackedWorkspace& packed_workspace() {
    static thread_local PackedWorkspace workspace;
    return workspace;
}

void zero_panel(Scalar* C, size_t ldc, size_t rows, size_t cols) {
    #pragma omp for schedule(static) nowait
    for (size_t i = 0; i < rows; ++i) {
        Scalar* Ci = C + i * ldc;
        for (size_t j = 0; j < cols; ++j) {
            Ci[j] = 0.0f;
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_full_avx2_8x8(const Scalar* packed_A,
                                      const Scalar* packed_B,
                                      Scalar* C,
                                      size_t ldc,
                                      size_t Kc) {
    Scalar* c0p = C + 0 * ldc;
    Scalar* c1p = C + 1 * ldc;
    Scalar* c2p = C + 2 * ldc;
    Scalar* c3p = C + 3 * ldc;
    Scalar* c4p = C + 4 * ldc;
    Scalar* c5p = C + 5 * ldc;
    Scalar* c6p = C + 6 * ldc;
    Scalar* c7p = C + 7 * ldc;

    __m256 c0 = _mm256_loadu_ps(c0p);
    __m256 c1 = _mm256_loadu_ps(c1p);
    __m256 c2 = _mm256_loadu_ps(c2p);
    __m256 c3 = _mm256_loadu_ps(c3p);
    __m256 c4 = _mm256_loadu_ps(c4p);
    __m256 c5 = _mm256_loadu_ps(c5p);
    __m256 c6 = _mm256_loadu_ps(c6p);
    __m256 c7 = _mm256_loadu_ps(c7p);

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b = _mm256_loadu_ps(b_ptr);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 0), b, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 1), b, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 2), b, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 3), b, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 4), b, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 5), b, c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 6), b, c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 7), b, c7);
        a_ptr += kKernel8x8.mr;
        b_ptr += kKernel8x8.nr;
    }

    _mm256_storeu_ps(c0p, c0);
    _mm256_storeu_ps(c1p, c1);
    _mm256_storeu_ps(c2p, c2);
    _mm256_storeu_ps(c3p, c3);
    _mm256_storeu_ps(c4p, c4);
    _mm256_storeu_ps(c5p, c5);
    _mm256_storeu_ps(c6p, c6);
    _mm256_storeu_ps(c7p, c7);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_fringe_avx2_8x8(const Scalar* packed_A,
                                        const Scalar* packed_B,
                                        Scalar* C,
                                        size_t ldc,
                                        size_t Kc,
                                        size_t rows,
                                        size_t cols) {
    const bool full_cols = cols == kKernel8x8.nr;
    const __m256i col_mask = full_cols
        ? _mm256_setzero_si256()
        : _mm256_set_epi32(
              cols > 7 ? -1 : 0,
              cols > 6 ? -1 : 0,
              cols > 5 ? -1 : 0,
              cols > 4 ? -1 : 0,
              cols > 3 ? -1 : 0,
              cols > 2 ? -1 : 0,
              cols > 1 ? -1 : 0,
              cols > 0 ? -1 : 0);

    __m256 c[8];
    for (size_t i = 0; i < kKernel8x8.mr; ++i) {
        if (i < rows) {
            Scalar* Ci = C + i * ldc;
            c[i] = full_cols ? _mm256_loadu_ps(Ci) : _mm256_maskload_ps(Ci, col_mask);
        } else {
            c[i] = _mm256_setzero_ps();
        }
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b = _mm256_loadu_ps(b_ptr);
        for (size_t i = 0; i < kKernel8x8.mr; ++i) {
            c[i] = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + i), b, c[i]);
        }
        a_ptr += kKernel8x8.mr;
        b_ptr += kKernel8x8.nr;
    }

    for (size_t i = 0; i < rows; ++i) {
        Scalar* Ci = C + i * ldc;
        if (full_cols) {
            _mm256_storeu_ps(Ci, c[i]);
        } else {
            _mm256_maskstore_ps(Ci, col_mask, c[i]);
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_full_avx2_12x8(const Scalar* packed_A,
                                       const Scalar* packed_B,
                                       Scalar* C,
                                       size_t ldc,
                                       size_t Kc) {
    __m256 c[12];
    for (size_t i = 0; i < 12; ++i) {
        c[i] = _mm256_loadu_ps(C + i * ldc);
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b = _mm256_loadu_ps(b_ptr);
        for (size_t i = 0; i < 12; ++i) {
            c[i] = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + i), b, c[i]);
        }
        a_ptr += 12;
        b_ptr += 8;
    }

    for (size_t i = 0; i < 12; ++i) {
        _mm256_storeu_ps(C + i * ldc, c[i]);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_fringe_avx2_12x8(const Scalar* packed_A,
                                         const Scalar* packed_B,
                                         Scalar* C,
                                         size_t ldc,
                                         size_t Kc,
                                         size_t rows,
                                         size_t cols) {
    const bool full_cols = cols == kKernel12x8.nr;
    const __m256i col_mask = full_cols
        ? _mm256_setzero_si256()
        : _mm256_set_epi32(
              cols > 7 ? -1 : 0,
              cols > 6 ? -1 : 0,
              cols > 5 ? -1 : 0,
              cols > 4 ? -1 : 0,
              cols > 3 ? -1 : 0,
              cols > 2 ? -1 : 0,
              cols > 1 ? -1 : 0,
              cols > 0 ? -1 : 0);

    __m256 c[12];
    for (size_t i = 0; i < 12; ++i) {
        if (i < rows) {
            Scalar* Ci = C + i * ldc;
            c[i] = full_cols ? _mm256_loadu_ps(Ci) : _mm256_maskload_ps(Ci, col_mask);
        } else {
            c[i] = _mm256_setzero_ps();
        }
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b = _mm256_loadu_ps(b_ptr);
        for (size_t i = 0; i < 12; ++i) {
            c[i] = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + i), b, c[i]);
        }
        a_ptr += 12;
        b_ptr += 8;
    }

    for (size_t i = 0; i < rows; ++i) {
        Scalar* Ci = C + i * ldc;
        if (full_cols) {
            _mm256_storeu_ps(Ci, c[i]);
        } else {
            _mm256_maskstore_ps(Ci, col_mask, c[i]);
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_full_avx2_13x8(const Scalar* packed_A,
                                       const Scalar* packed_B,
                                       Scalar* C,
                                       size_t ldc,
                                       size_t Kc) {
    __m256 c[13];
    for (size_t i = 0; i < 13; ++i) {
        c[i] = _mm256_loadu_ps(C + i * ldc);
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b = _mm256_loadu_ps(b_ptr);
        for (size_t i = 0; i < 13; ++i) {
            c[i] = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + i), b, c[i]);
        }
        a_ptr += 13;
        b_ptr += 8;
    }

    for (size_t i = 0; i < 13; ++i) {
        _mm256_storeu_ps(C + i * ldc, c[i]);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_fringe_avx2_13x8(const Scalar* packed_A,
                                         const Scalar* packed_B,
                                         Scalar* C,
                                         size_t ldc,
                                         size_t Kc,
                                         size_t rows,
                                         size_t cols) {
    const bool full_cols = cols == kKernel13x8.nr;
    const __m256i col_mask = full_cols
        ? _mm256_setzero_si256()
        : _mm256_set_epi32(
              cols > 7 ? -1 : 0,
              cols > 6 ? -1 : 0,
              cols > 5 ? -1 : 0,
              cols > 4 ? -1 : 0,
              cols > 3 ? -1 : 0,
              cols > 2 ? -1 : 0,
              cols > 1 ? -1 : 0,
              cols > 0 ? -1 : 0);

    __m256 c[13];
    for (size_t i = 0; i < 13; ++i) {
        if (i < rows) {
            Scalar* Ci = C + i * ldc;
            c[i] = full_cols ? _mm256_loadu_ps(Ci) : _mm256_maskload_ps(Ci, col_mask);
        } else {
            c[i] = _mm256_setzero_ps();
        }
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b = _mm256_loadu_ps(b_ptr);
        for (size_t i = 0; i < 13; ++i) {
            c[i] = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + i), b, c[i]);
        }
        a_ptr += 13;
        b_ptr += 8;
    }

    for (size_t i = 0; i < rows; ++i) {
        Scalar* Ci = C + i * ldc;
        if (full_cols) {
            _mm256_storeu_ps(Ci, c[i]);
        } else {
            _mm256_maskstore_ps(Ci, col_mask, c[i]);
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_full_avx2_4x16(const Scalar* packed_A,
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

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(b_ptr + 0);
        const __m256 b1 = _mm256_loadu_ps(b_ptr + 8);
        c00 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 0), b0, c00);
        c01 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 0), b1, c01);
        c10 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 1), b0, c10);
        c11 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 1), b1, c11);
        c20 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 2), b0, c20);
        c21 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 2), b1, c21);
        c30 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 3), b0, c30);
        c31 = _mm256_fmadd_ps(_mm256_broadcast_ss(a_ptr + 3), b1, c31);
        a_ptr += kKernel4x16.mr;
        b_ptr += kKernel4x16.nr;
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
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_full_avx2_5x16(const Scalar* packed_A,
                                       const Scalar* packed_B,
                                       Scalar* C,
                                       size_t ldc,
                                       size_t Kc) {
    __m256 c_lo[5];
    __m256 c_hi[5];
    for (size_t i = 0; i < 5; ++i) {
        c_lo[i] = _mm256_loadu_ps(C + i * ldc + 0);
        c_hi[i] = _mm256_loadu_ps(C + i * ldc + 8);
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(b_ptr + 0);
        const __m256 b1 = _mm256_loadu_ps(b_ptr + 8);
        for (size_t i = 0; i < 5; ++i) {
            const __m256 a = _mm256_broadcast_ss(a_ptr + i);
            c_lo[i] = _mm256_fmadd_ps(a, b0, c_lo[i]);
            c_hi[i] = _mm256_fmadd_ps(a, b1, c_hi[i]);
        }
        a_ptr += 5;
        b_ptr += 16;
    }

    for (size_t i = 0; i < 5; ++i) {
        _mm256_storeu_ps(C + i * ldc + 0, c_lo[i]);
        _mm256_storeu_ps(C + i * ldc + 8, c_hi[i]);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_fringe_avx2_5x16(const Scalar* packed_A,
                                         const Scalar* packed_B,
                                         Scalar* C,
                                         size_t ldc,
                                         size_t Kc,
                                         size_t rows,
                                         size_t cols) {
    const size_t cols_lo = std::min<size_t>(8, cols);
    const size_t cols_hi = cols > 8 ? cols - 8 : 0;
    const bool full_lo = cols_lo == 8;
    const bool full_hi = cols_hi == 8;
    const __m256i mask_lo = full_lo
        ? _mm256_setzero_si256()
        : _mm256_set_epi32(
              cols_lo > 7 ? -1 : 0,
              cols_lo > 6 ? -1 : 0,
              cols_lo > 5 ? -1 : 0,
              cols_lo > 4 ? -1 : 0,
              cols_lo > 3 ? -1 : 0,
              cols_lo > 2 ? -1 : 0,
              cols_lo > 1 ? -1 : 0,
              cols_lo > 0 ? -1 : 0);
    const __m256i mask_hi = full_hi
        ? _mm256_setzero_si256()
        : _mm256_set_epi32(
              cols_hi > 7 ? -1 : 0,
              cols_hi > 6 ? -1 : 0,
              cols_hi > 5 ? -1 : 0,
              cols_hi > 4 ? -1 : 0,
              cols_hi > 3 ? -1 : 0,
              cols_hi > 2 ? -1 : 0,
              cols_hi > 1 ? -1 : 0,
              cols_hi > 0 ? -1 : 0);

    __m256 c_lo[5];
    __m256 c_hi[5];
    for (size_t i = 0; i < 5; ++i) {
        if (i < rows) {
            Scalar* Ci = C + i * ldc;
            c_lo[i] = full_lo ? _mm256_loadu_ps(Ci) : _mm256_maskload_ps(Ci, mask_lo);
            c_hi[i] = cols_hi == 0
                ? _mm256_setzero_ps()
                : (full_hi ? _mm256_loadu_ps(Ci + 8) : _mm256_maskload_ps(Ci + 8, mask_hi));
        } else {
            c_lo[i] = _mm256_setzero_ps();
            c_hi[i] = _mm256_setzero_ps();
        }
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(b_ptr + 0);
        const __m256 b1 = _mm256_loadu_ps(b_ptr + 8);
        for (size_t i = 0; i < 5; ++i) {
            const __m256 a = _mm256_broadcast_ss(a_ptr + i);
            c_lo[i] = _mm256_fmadd_ps(a, b0, c_lo[i]);
            c_hi[i] = _mm256_fmadd_ps(a, b1, c_hi[i]);
        }
        a_ptr += 5;
        b_ptr += 16;
    }

    for (size_t i = 0; i < rows; ++i) {
        Scalar* Ci = C + i * ldc;
        if (full_lo) {
            _mm256_storeu_ps(Ci, c_lo[i]);
        } else {
            _mm256_maskstore_ps(Ci, mask_lo, c_lo[i]);
        }
        if (cols_hi == 0) {
            continue;
        }
        if (full_hi) {
            _mm256_storeu_ps(Ci + 8, c_hi[i]);
        } else {
            _mm256_maskstore_ps(Ci + 8, mask_hi, c_hi[i]);
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_full_avx2_6x16(const Scalar* packed_A,
                                       const Scalar* packed_B,
                                       Scalar* C,
                                       size_t ldc,
                                       size_t Kc) {
    __m256 c_lo[6];
    __m256 c_hi[6];
    for (size_t i = 0; i < 6; ++i) {
        c_lo[i] = _mm256_loadu_ps(C + i * ldc + 0);
        c_hi[i] = _mm256_loadu_ps(C + i * ldc + 8);
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(b_ptr + 0);
        const __m256 b1 = _mm256_loadu_ps(b_ptr + 8);
        for (size_t i = 0; i < 6; ++i) {
            const __m256 a = _mm256_broadcast_ss(a_ptr + i);
            c_lo[i] = _mm256_fmadd_ps(a, b0, c_lo[i]);
            c_hi[i] = _mm256_fmadd_ps(a, b1, c_hi[i]);
        }
        a_ptr += 6;
        b_ptr += 16;
    }

    for (size_t i = 0; i < 6; ++i) {
        _mm256_storeu_ps(C + i * ldc + 0, c_lo[i]);
        _mm256_storeu_ps(C + i * ldc + 8, c_hi[i]);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_fringe_avx2_6x16(const Scalar* packed_A,
                                         const Scalar* packed_B,
                                         Scalar* C,
                                         size_t ldc,
                                         size_t Kc,
                                         size_t rows,
                                         size_t cols) {
    const size_t cols_lo = std::min<size_t>(8, cols);
    const size_t cols_hi = cols > 8 ? cols - 8 : 0;
    const bool full_lo = cols_lo == 8;
    const bool full_hi = cols_hi == 8;
    const __m256i mask_lo = full_lo
        ? _mm256_setzero_si256()
        : _mm256_set_epi32(
              cols_lo > 7 ? -1 : 0,
              cols_lo > 6 ? -1 : 0,
              cols_lo > 5 ? -1 : 0,
              cols_lo > 4 ? -1 : 0,
              cols_lo > 3 ? -1 : 0,
              cols_lo > 2 ? -1 : 0,
              cols_lo > 1 ? -1 : 0,
              cols_lo > 0 ? -1 : 0);
    const __m256i mask_hi = full_hi
        ? _mm256_setzero_si256()
        : _mm256_set_epi32(
              cols_hi > 7 ? -1 : 0,
              cols_hi > 6 ? -1 : 0,
              cols_hi > 5 ? -1 : 0,
              cols_hi > 4 ? -1 : 0,
              cols_hi > 3 ? -1 : 0,
              cols_hi > 2 ? -1 : 0,
              cols_hi > 1 ? -1 : 0,
              cols_hi > 0 ? -1 : 0);

    __m256 c_lo[6];
    __m256 c_hi[6];
    for (size_t i = 0; i < 6; ++i) {
        if (i < rows) {
            Scalar* Ci = C + i * ldc;
            c_lo[i] = full_lo ? _mm256_loadu_ps(Ci) : _mm256_maskload_ps(Ci, mask_lo);
            c_hi[i] = cols_hi == 0
                ? _mm256_setzero_ps()
                : (full_hi ? _mm256_loadu_ps(Ci + 8) : _mm256_maskload_ps(Ci + 8, mask_hi));
        } else {
            c_lo[i] = _mm256_setzero_ps();
            c_hi[i] = _mm256_setzero_ps();
        }
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(b_ptr + 0);
        const __m256 b1 = _mm256_loadu_ps(b_ptr + 8);
        for (size_t i = 0; i < 6; ++i) {
            const __m256 a = _mm256_broadcast_ss(a_ptr + i);
            c_lo[i] = _mm256_fmadd_ps(a, b0, c_lo[i]);
            c_hi[i] = _mm256_fmadd_ps(a, b1, c_hi[i]);
        }
        a_ptr += 6;
        b_ptr += 16;
    }

    for (size_t i = 0; i < rows; ++i) {
        Scalar* Ci = C + i * ldc;
        if (full_lo) {
            _mm256_storeu_ps(Ci, c_lo[i]);
        } else {
            _mm256_maskstore_ps(Ci, mask_lo, c_lo[i]);
        }
        if (cols_hi == 0) {
            continue;
        }
        if (full_hi) {
            _mm256_storeu_ps(Ci + 8, c_hi[i]);
        } else {
            _mm256_maskstore_ps(Ci + 8, mask_hi, c_hi[i]);
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
inline void microkernel_fringe_avx2_4x16(const Scalar* packed_A,
                                         const Scalar* packed_B,
                                         Scalar* C,
                                         size_t ldc,
                                         size_t Kc,
                                         size_t rows,
                                         size_t cols) {
    const size_t cols_lo = std::min<size_t>(8, cols);
    const size_t cols_hi = cols > 8 ? cols - 8 : 0;
    const bool full_lo = cols_lo == 8;
    const bool full_hi = cols_hi == 8;
    const __m256i mask_lo = full_lo
        ? _mm256_setzero_si256()
        : _mm256_set_epi32(
              cols_lo > 7 ? -1 : 0,
              cols_lo > 6 ? -1 : 0,
              cols_lo > 5 ? -1 : 0,
              cols_lo > 4 ? -1 : 0,
              cols_lo > 3 ? -1 : 0,
              cols_lo > 2 ? -1 : 0,
              cols_lo > 1 ? -1 : 0,
              cols_lo > 0 ? -1 : 0);
    const __m256i mask_hi = full_hi
        ? _mm256_setzero_si256()
        : _mm256_set_epi32(
              cols_hi > 7 ? -1 : 0,
              cols_hi > 6 ? -1 : 0,
              cols_hi > 5 ? -1 : 0,
              cols_hi > 4 ? -1 : 0,
              cols_hi > 3 ? -1 : 0,
              cols_hi > 2 ? -1 : 0,
              cols_hi > 1 ? -1 : 0,
              cols_hi > 0 ? -1 : 0);

    __m256 c_lo[4];
    __m256 c_hi[4];
    for (size_t i = 0; i < kKernel4x16.mr; ++i) {
        if (i < rows) {
            Scalar* Ci = C + i * ldc;
            c_lo[i] = full_lo ? _mm256_loadu_ps(Ci) : _mm256_maskload_ps(Ci, mask_lo);
            c_hi[i] = cols_hi == 0
                ? _mm256_setzero_ps()
                : (full_hi ? _mm256_loadu_ps(Ci + 8) : _mm256_maskload_ps(Ci + 8, mask_hi));
        } else {
            c_lo[i] = _mm256_setzero_ps();
            c_hi[i] = _mm256_setzero_ps();
        }
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(b_ptr + 0);
        const __m256 b1 = _mm256_loadu_ps(b_ptr + 8);
        for (size_t i = 0; i < kKernel4x16.mr; ++i) {
            const __m256 a = _mm256_broadcast_ss(a_ptr + i);
            c_lo[i] = _mm256_fmadd_ps(a, b0, c_lo[i]);
            c_hi[i] = _mm256_fmadd_ps(a, b1, c_hi[i]);
        }
        a_ptr += kKernel4x16.mr;
        b_ptr += kKernel4x16.nr;
    }

    for (size_t i = 0; i < rows; ++i) {
        Scalar* Ci = C + i * ldc;
        if (full_lo) {
            _mm256_storeu_ps(Ci, c_lo[i]);
        } else {
            _mm256_maskstore_ps(Ci, mask_lo, c_lo[i]);
        }
        if (cols_hi == 0) {
            continue;
        }
        if (full_hi) {
            _mm256_storeu_ps(Ci + 8, c_hi[i]);
        } else {
            _mm256_maskstore_ps(Ci + 8, mask_hi, c_hi[i]);
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif
inline void microkernel_full_avx512_mx16(const Scalar* packed_A,
                                         const Scalar* packed_B,
                                         Scalar* C,
                                         size_t ldc,
                                         size_t Kc,
                                         size_t mr) {
    __m512 c[20];
    for (size_t i = 0; i < mr; ++i) {
        c[i] = _mm512_loadu_ps(C + i * ldc);
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m512 b = _mm512_loadu_ps(b_ptr);
        for (size_t i = 0; i < mr; ++i) {
            c[i] = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[i]), b, c[i]);
        }
        a_ptr += mr;
        b_ptr += 16;
    }

    for (size_t i = 0; i < mr; ++i) {
        _mm512_storeu_ps(C + i * ldc, c[i]);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif
inline void microkernel_fringe_avx512_mx16(const Scalar* packed_A,
                                           const Scalar* packed_B,
                                           Scalar* C,
                                           size_t ldc,
                                           size_t Kc,
                                           size_t mr,
                                           size_t rows,
                                           size_t cols) {
    const __mmask16 col_mask = cols >= 16
        ? static_cast<__mmask16>(0xFFFFu)
        : static_cast<__mmask16>((1u << cols) - 1u);
    const bool full_cols = col_mask == static_cast<__mmask16>(0xFFFFu);

    __m512 c[20];
    for (size_t i = 0; i < mr; ++i) {
        if (i < rows) {
            Scalar* Ci = C + i * ldc;
            c[i] = full_cols ? _mm512_loadu_ps(Ci) : _mm512_maskz_loadu_ps(col_mask, Ci);
        } else {
            c[i] = _mm512_setzero_ps();
        }
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m512 b = _mm512_loadu_ps(b_ptr);
        for (size_t i = 0; i < mr; ++i) {
            c[i] = _mm512_fmadd_ps(_mm512_set1_ps(a_ptr[i]), b, c[i]);
        }
        a_ptr += mr;
        b_ptr += 16;
    }

    for (size_t i = 0; i < rows; ++i) {
        Scalar* Ci = C + i * ldc;
        if (full_cols) {
            _mm512_storeu_ps(Ci, c[i]);
        } else {
            _mm512_mask_storeu_ps(Ci, col_mask, c[i]);
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif
inline void microkernel_full_avx512_mx32(const Scalar* packed_A,
                                         const Scalar* packed_B,
                                         Scalar* C,
                                         size_t ldc,
                                         size_t Kc,
                                         size_t mr) {
    __m512 c_lo[12];
    __m512 c_hi[12];
    for (size_t i = 0; i < mr; ++i) {
        c_lo[i] = _mm512_loadu_ps(C + i * ldc);
        c_hi[i] = _mm512_loadu_ps(C + i * ldc + 16);
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m512 b_lo = _mm512_loadu_ps(b_ptr);
        const __m512 b_hi = _mm512_loadu_ps(b_ptr + 16);
        for (size_t i = 0; i < mr; ++i) {
            const __m512 a = _mm512_set1_ps(a_ptr[i]);
            c_lo[i] = _mm512_fmadd_ps(a, b_lo, c_lo[i]);
            c_hi[i] = _mm512_fmadd_ps(a, b_hi, c_hi[i]);
        }
        a_ptr += mr;
        b_ptr += 32;
    }

    for (size_t i = 0; i < mr; ++i) {
        _mm512_storeu_ps(C + i * ldc, c_lo[i]);
        _mm512_storeu_ps(C + i * ldc + 16, c_hi[i]);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f")))
#endif
inline void microkernel_fringe_avx512_mx32(const Scalar* packed_A,
                                           const Scalar* packed_B,
                                           Scalar* C,
                                           size_t ldc,
                                           size_t Kc,
                                           size_t mr,
                                           size_t rows,
                                           size_t cols) {
    const size_t cols_lo = std::min<size_t>(16, cols);
    const size_t cols_hi = cols > 16 ? cols - 16 : 0;
    const __mmask16 mask_lo = cols_lo == 16
        ? static_cast<__mmask16>(0xFFFFu)
        : static_cast<__mmask16>((1u << cols_lo) - 1u);
    const __mmask16 mask_hi = cols_hi >= 16
        ? static_cast<__mmask16>(0xFFFFu)
        : static_cast<__mmask16>((1u << cols_hi) - 1u);
    const bool full_lo = mask_lo == static_cast<__mmask16>(0xFFFFu);
    const bool full_hi = mask_hi == static_cast<__mmask16>(0xFFFFu);

    __m512 c_lo[12];
    __m512 c_hi[12];
    for (size_t i = 0; i < mr; ++i) {
        if (i < rows) {
            Scalar* Ci = C + i * ldc;
            c_lo[i] = full_lo ? _mm512_loadu_ps(Ci) : _mm512_maskz_loadu_ps(mask_lo, Ci);
            c_hi[i] = cols_hi == 0
                ? _mm512_setzero_ps()
                : (full_hi ? _mm512_loadu_ps(Ci + 16) : _mm512_maskz_loadu_ps(mask_hi, Ci + 16));
        } else {
            c_lo[i] = _mm512_setzero_ps();
            c_hi[i] = _mm512_setzero_ps();
        }
    }

    const Scalar* a_ptr = packed_A;
    const Scalar* b_ptr = packed_B;
    for (size_t k = 0; k < Kc; ++k) {
        const __m512 b_lo = _mm512_loadu_ps(b_ptr);
        const __m512 b_hi = _mm512_loadu_ps(b_ptr + 16);
        for (size_t i = 0; i < mr; ++i) {
            const __m512 a = _mm512_set1_ps(a_ptr[i]);
            c_lo[i] = _mm512_fmadd_ps(a, b_lo, c_lo[i]);
            c_hi[i] = _mm512_fmadd_ps(a, b_hi, c_hi[i]);
        }
        a_ptr += mr;
        b_ptr += 32;
    }

    for (size_t i = 0; i < rows; ++i) {
        Scalar* Ci = C + i * ldc;
        if (full_lo) {
            _mm512_storeu_ps(Ci, c_lo[i]);
        } else {
            _mm512_mask_storeu_ps(Ci, mask_lo, c_lo[i]);
        }
        if (cols_hi == 0) {
            continue;
        }
        if (full_hi) {
            _mm512_storeu_ps(Ci + 16, c_hi[i]);
        } else {
            _mm512_mask_storeu_ps(Ci + 16, mask_hi, c_hi[i]);
        }
    }
}

void run_selected_microkernel(const KernelShape& kernel,
                              const Scalar* packed_A,
                              const Scalar* packed_B,
                              Scalar* C,
                              size_t ldc,
                              size_t Kc,
                              size_t rows,
                              size_t cols) {
    if (kernel.isa == KernelIsa::Avx512) {
        if (!cpu_supports_avx512f()) {
            throw std::runtime_error(
                "AVX-512 GotoBLAS micro-kernel requires AVX-512F support");
        }
        if ((kernel.mr == kKernelAvx5124x16.mr ||
             kernel.mr == kKernelAvx5128x16.mr ||
             kernel.mr == kKernelAvx51214x16.mr ||
             kernel.mr == kKernelAvx51216x16.mr ||
             kernel.mr == kKernelAvx51218x16.mr ||
             kernel.mr == kKernelAvx51220x16.mr) &&
            kernel.nr == 16) {
            if (rows == kernel.mr && cols == kernel.nr) {
                microkernel_full_avx512_mx16(packed_A, packed_B, C, ldc, Kc, kernel.mr);
            } else {
                microkernel_fringe_avx512_mx16(
                    packed_A, packed_B, C, ldc, Kc, kernel.mr, rows, cols);
            }
            return;
        }
        if ((kernel.mr == kKernelAvx5124x32.mr ||
             kernel.mr == kKernelAvx5126x32.mr ||
             kernel.mr == kKernelAvx5128x32.mr ||
             kernel.mr == kKernelAvx51210x32.mr ||
             kernel.mr == kKernelAvx51212x32.mr) &&
            kernel.nr == 32) {
            if (rows == kernel.mr && cols == kernel.nr) {
                microkernel_full_avx512_mx32(packed_A, packed_B, C, ldc, Kc, kernel.mr);
            } else {
                microkernel_fringe_avx512_mx32(
                    packed_A, packed_B, C, ldc, Kc, kernel.mr, rows, cols);
            }
            return;
        }
        throw std::invalid_argument("Unsupported AVX-512 GotoBLAS micro-kernel shape");
    }

    if (kernel.mr == kKernel8x8.mr && kernel.nr == kKernel8x8.nr) {
        if (rows == kernel.mr && cols == kernel.nr) {
            microkernel_full_avx2_8x8(packed_A, packed_B, C, ldc, Kc);
        } else {
            microkernel_fringe_avx2_8x8(packed_A, packed_B, C, ldc, Kc, rows, cols);
        }
        return;
    }
    if (kernel.mr == kKernel12x8.mr && kernel.nr == kKernel12x8.nr) {
        if (rows == kernel.mr && cols == kernel.nr) {
            microkernel_full_avx2_12x8(packed_A, packed_B, C, ldc, Kc);
        } else {
            microkernel_fringe_avx2_12x8(packed_A, packed_B, C, ldc, Kc, rows, cols);
        }
        return;
    }
    if (kernel.mr == kKernel13x8.mr && kernel.nr == kKernel13x8.nr) {
        if (rows == kernel.mr && cols == kernel.nr) {
            microkernel_full_avx2_13x8(packed_A, packed_B, C, ldc, Kc);
        } else {
            microkernel_fringe_avx2_13x8(packed_A, packed_B, C, ldc, Kc, rows, cols);
        }
        return;
    }
    if (kernel.mr == kKernel4x16.mr && kernel.nr == kKernel4x16.nr) {
        if (rows == kernel.mr && cols == kernel.nr) {
            microkernel_full_avx2_4x16(packed_A, packed_B, C, ldc, Kc);
        } else {
            microkernel_fringe_avx2_4x16(packed_A, packed_B, C, ldc, Kc, rows, cols);
        }
        return;
    }
    if (kernel.mr == kKernel5x16.mr && kernel.nr == kKernel5x16.nr) {
        if (rows == kernel.mr && cols == kernel.nr) {
            microkernel_full_avx2_5x16(packed_A, packed_B, C, ldc, Kc);
        } else {
            microkernel_fringe_avx2_5x16(packed_A, packed_B, C, ldc, Kc, rows, cols);
        }
        return;
    }
    if (kernel.mr == kKernel6x16.mr && kernel.nr == kKernel6x16.nr) {
        if (rows == kernel.mr && cols == kernel.nr) {
            microkernel_full_avx2_6x16(packed_A, packed_B, C, ldc, Kc);
        } else {
            microkernel_fringe_avx2_6x16(packed_A, packed_B, C, ldc, Kc, rows, cols);
        }
        return;
    }
    throw std::invalid_argument("Unsupported GotoBLAS micro-kernel shape");
}

void pack_B_block(const Scalar* B_block,
                  Scalar* packed_B,
                  size_t ldb,
                  size_t Kc,
                  size_t Nc,
                  const KernelShape& kernel) {
    const size_t nr_panels = (Nc + kernel.nr - 1) / kernel.nr;

    for (size_t jr = 0; jr < nr_panels; ++jr) {
        const size_t jc_inner = jr * kernel.nr;
        const size_t cols = minz(kernel.nr, Nc - jc_inner);
        pack_b_micro_panel(
            B_block + jc_inner,
            packed_B + jr * Kc * kernel.nr,
            ldb,
            Kc,
            cols,
            kernel.nr);
    }
}

void pack_A_block(const Scalar* A_block,
                  Scalar* packed_A,
                  size_t lda,
                  size_t Mc,
                  size_t Kc,
                  const KernelShape& kernel) {
    const size_t mr_panels = (Mc + kernel.mr - 1) / kernel.mr;

    for (size_t ir = 0; ir < mr_panels; ++ir) {
        const size_t ic_inner = ir * kernel.mr;
        const size_t rows = minz(kernel.mr, Mc - ic_inner);
        pack_a_micro_panel(
            A_block + ic_inner * lda,
            packed_A + ir * Kc * kernel.mr,
            lda,
            Kc,
            rows,
            kernel.mr);
    }
}

void run_macro_kernel(const Scalar* packed_A,
                      const Scalar* packed_B,
                      Scalar* C_block,
                      size_t ldc,
                      size_t Mc,
                      size_t Nc,
                      size_t Kc,
                      const KernelShape& kernel) {
    const size_t mr_panels = (Mc + kernel.mr - 1) / kernel.mr;
    const size_t nr_panels = (Nc + kernel.nr - 1) / kernel.nr;

    for (size_t jr = 0; jr < nr_panels; ++jr) {
        for (size_t ir = 0; ir < mr_panels; ++ir) {
            const size_t jc_inner = jr * kernel.nr;
            const size_t ic_inner = ir * kernel.mr;
            const size_t cols = minz(kernel.nr, Nc - jc_inner);
            const size_t rows = minz(kernel.mr, Mc - ic_inner);

            const Scalar* packed_A_panel = packed_A + ir * Kc * kernel.mr;
            const Scalar* packed_B_panel = packed_B + jr * Kc * kernel.nr;
            Scalar* Ctile = C_block + ic_inner * ldc + jc_inner;
            run_selected_microkernel(kernel, packed_A_panel, packed_B_panel, Ctile, ldc, Kc, rows, cols);
        }
    }
}

// Compute per-thread scheduling granularity for the ic (row) dimension.
// Identical in spirit to GEMMMultiThread's compute_task_m but uses kernel.mr
// as the minimum so every task holds at least one full micro-kernel row panel.
// task_m ≤ Mc always, so packed_A (sized on Mc) is never overflowed.
inline size_t compute_goto_task_m(size_t M, size_t Mc, int nthreads, size_t mr) {
    constexpr size_t kTargetTasksPerThread = 2;
    if (nthreads <= 0 || M == 0) return Mc;
    const size_t min_task_m = mr > 0 ? mr : 1;
    const size_t target_tasks =
        static_cast<size_t>(nthreads) * kTargetTasksPerThread;
    const size_t rows_per_target = (M + target_tasks - 1) / target_tasks;
    const size_t clamped =
        rows_per_target < min_task_m ? min_task_m : rows_per_target;
    return clamped < Mc ? clamped : Mc;
}

void run_gotoblas_fixed(const Scalar* A,
                        const Scalar* B,
                        Scalar* C,
                        size_t M,
                        size_t N,
                        size_t K,
                        KernelIsa isa) {
    const KernelShape& kernel = current_kernel_shape(isa);
    const size_t Mc = current_mc_block_size();
    const size_t Nc = current_nc_block_size();
    const size_t Kc = current_kc_block_size();
    const size_t max_mr_panels = (Mc + kernel.mr - 1) / kernel.mr;
    const size_t max_nr_panels = (Nc + kernel.nr - 1) / kernel.nr;
    PackedWorkspace& shared_workspace = packed_workspace();
    Scalar* packed_B = shared_workspace.ensure_packed_b(max_nr_panels * Kc * kernel.nr);
    const size_t packed_A_elements = max_mr_panels * Kc * kernel.mr;

    #pragma omp parallel
    {
        PackedWorkspace& local_workspace = packed_workspace();
        Scalar* packed_A = local_workspace.ensure_packed_a(packed_A_elements);

        // task_m: scheduling granularity for the ic dimension, computed once
        // per parallel region.  task_m ≤ Mc so packed_A is never too small.
        // For small M this creates more tasks than mc_blocks would, keeping
        // threads busy.  For large M it saturates at Mc (original behavior).
        const int nthreads = omp_get_num_threads();
        const size_t task_m = compute_goto_task_m(M, Mc, nthreads, kernel.mr);
        const size_t task_blocks = (M + task_m - 1) / task_m;

        for (size_t jc = 0; jc < N; jc += Nc) {
            const size_t nc_cur = minz(Nc, N - jc);
            zero_panel(C + jc, N, M, nc_cur);

            for (size_t pc = 0; pc < K; pc += Kc) {
                const size_t kc_cur = minz(Kc, K - pc);
                // pack_B_block is done by a single thread; the implicit
                // barrier of omp single ensures all threads see the result.
                #pragma omp single
                {
                    pack_B_block(B + pc * N + jc, packed_B, N, kc_cur, nc_cur, kernel);
                }

                // Distribute finer-grained ic tasks among threads.
                // pack_A_block and run_macro_kernel remain per-task so each
                // thread owns exclusive C rows — no atomic/reduction needed.
                #pragma omp for schedule(static)
                for (size_t task_idx = 0; task_idx < task_blocks; ++task_idx) {
                    const size_t ic = task_idx * task_m;
                    const size_t mc_cur = minz(task_m, M - ic);
                    pack_A_block(A + ic * K + pc, packed_A, K, mc_cur, kc_cur, kernel);
                    run_macro_kernel(
                        packed_A,
                        packed_B,
                        C + ic * N + jc,
                        N,
                        mc_cur,
                        nc_cur,
                        kc_cur,
                        kernel);
                }
            }
        }
    }
}

}  // namespace

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

size_t gotoblas_default_mr() { return kDefaultMicroKernelRows; }
size_t gotoblas_default_nr() { return kDefaultMicroKernelCols; }
size_t gotoblas_default_mc() { return kCacheBlockM; }
size_t gotoblas_default_nc() { return kCacheBlockN; }
size_t gotoblas_default_kc() { return kCacheBlockK; }

const char* gotoblas_current_kernel_name() {
    const KernelIsa isa = current_impl() == MatmulImpl::OmpGotoBlasAvx512
        ? KernelIsa::Avx512
        : KernelIsa::Avx2;
    return current_kernel_shape(isa).name;
}

size_t gotoblas_current_mr() {
    const KernelIsa isa = current_impl() == MatmulImpl::OmpGotoBlasAvx512
        ? KernelIsa::Avx512
        : KernelIsa::Avx2;
    return current_kernel_shape(isa).mr;
}

size_t gotoblas_current_nr() {
    const KernelIsa isa = current_impl() == MatmulImpl::OmpGotoBlasAvx512
        ? KernelIsa::Avx512
        : KernelIsa::Avx2;
    return current_kernel_shape(isa).nr;
}

bool gotoblas_try_get_kernel_shape(const char* shape_name, size_t* mr, size_t* nr) {
    const KernelShape* shape = find_kernel_shape(shape_name);
    if (shape == nullptr) {
        return false;
    }
    if (mr != nullptr) {
        *mr = shape->mr;
    }
    if (nr != nullptr) {
        *nr = shape->nr;
    }
    return true;
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

void gotoblas_microkernel_fixed(const Scalar* packed_A,
                                const Scalar* packed_B,
                                Scalar* C,
                                size_t ldc,
                                size_t Kc,
                                size_t rows,
                                size_t cols) {
    run_selected_microkernel(current_kernel_shape(), packed_A, packed_B, C, ldc, Kc, rows, cols);
}

void gotoblas_microkernel_for_shape(const char* shape_name,
                                    const Scalar* packed_A,
                                    const Scalar* packed_B,
                                    Scalar* C,
                                    size_t ldc,
                                    size_t Kc,
                                    size_t rows,
                                    size_t cols) {
    const KernelShape* kernel = find_kernel_shape(shape_name);
    if (kernel == nullptr) {
        throw std::invalid_argument("Unknown GotoBLAS micro-kernel shape");
    }
    run_selected_microkernel(*kernel, packed_A, packed_B, C, ldc, Kc, rows, cols);
}

void sgemm_omp_gotoblas_avx2(const Scalar* A,
                             const Scalar* B,
                             Scalar* C,
                             size_t M,
                             size_t N,
                             size_t K) {
    if (!cpu_supports_avx2_fma()) {
        throw std::runtime_error(
            "omp_gotoblas_avx2 requires AVX2/FMA support; no legacy fallback is retained");
    }

    run_gotoblas_fixed(A, B, C, M, N, K, KernelIsa::Avx2);
}

void sgemm_omp_gotoblas_avx512(const Scalar* A,
                               const Scalar* B,
                               Scalar* C,
                               size_t M,
                               size_t N,
                               size_t K) {
    if (!cpu_supports_avx512f()) {
        throw std::runtime_error(
            "omp_gotoblas_avx512 requires AVX-512F support; no AVX2 alias fallback is used");
    }

    run_gotoblas_fixed(A, B, C, M, N, K, KernelIsa::Avx512);
}

}  // namespace gemm
