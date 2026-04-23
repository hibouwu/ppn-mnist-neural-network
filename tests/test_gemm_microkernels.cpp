#include "gemm/matmul_internal.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool nearly_equal(Scalar a, Scalar b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

bool test_pack_a_micro_panel() {
    const size_t lda = 7;
    const size_t kc = 3;
    const size_t rows = 2;
    const size_t mr = 4;

    std::vector<Scalar> A{
        1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14,
    };
    std::vector<Scalar> packed(kc * mr, -1.0f);

    gemm::pack_a_micro_panel(A.data(), packed.data(), lda, kc, rows, mr);

    const std::vector<Scalar> expected{
        1, 8, 0, 0,
        2, 9, 0, 0,
        3, 10, 0, 0,
    };

    if (packed != expected) {
        std::cerr << "pack_a_micro_panel failed\n";
        return false;
    }
    return true;
}

bool test_pack_b_micro_panel() {
    const size_t ldb = 6;
    const size_t kc = 3;
    const size_t cols = 3;
    const size_t nr = 4;

    std::vector<Scalar> B{
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18,
    };
    std::vector<Scalar> packed(kc * nr, -1.0f);

    gemm::pack_b_micro_panel(B.data(), packed.data(), ldb, kc, cols, nr);

    const std::vector<Scalar> expected{
        1, 2, 3, 0,
        7, 8, 9, 0,
        13, 14, 15, 0,
    };

    if (packed != expected) {
        std::cerr << "pack_b_micro_panel failed\n";
        return false;
    }
    return true;
}

void reference_kernel(const std::vector<Scalar>& packed_a,
                      const std::vector<Scalar>& packed_b,
                      std::vector<Scalar>& c,
                      size_t mr,
                      size_t nr,
                      size_t packed_mr,
                      size_t packed_nr,
                      size_t ldc,
                      size_t kc) {
    for (size_t i = 0; i < mr; ++i) {
        for (size_t j = 0; j < nr; ++j) {
            Scalar acc = c[i * ldc + j];
            for (size_t k = 0; k < kc; ++k) {
                acc += packed_a[k * packed_mr + i]
                    * packed_b[k * packed_nr + j];
            }
            c[i * ldc + j] = acc;
        }
    }
}

bool test_microkernel_shape_full_tile(const char* shape_name) {
    if (!gemm::cpu_supports_avx2_fma()) {
        std::cout << "Skipping fixed micro-kernel test (AVX2/FMA unavailable)\n";
        return true;
    }

    size_t mr = 0;
    size_t nr = 0;
    if (!gemm::gotoblas_try_get_kernel_shape(shape_name, &mr, &nr)) {
        std::cerr << "Unknown micro-kernel shape: " << shape_name << "\n";
        return false;
    }
    const size_t kc = 5;
    const size_t ldc = nr + 3;

    std::vector<Scalar> packed_a(kc * mr);
    std::vector<Scalar> packed_b(kc * nr);
    std::vector<Scalar> c_test(mr * ldc, 0.25f);
    std::vector<Scalar> c_ref(mr * ldc, 0.25f);

    for (size_t k = 0; k < kc; ++k) {
        for (size_t i = 0; i < mr; ++i) {
            packed_a[k * mr + i] =
                static_cast<Scalar>((static_cast<int>(i) - 3) * 0.5f + static_cast<float>(k));
        }
        for (size_t j = 0; j < nr; ++j) {
            packed_b[k * nr + j] =
                static_cast<Scalar>((static_cast<int>(j) - 2) * 0.25f - static_cast<float>(k) * 0.1f);
        }
    }

    reference_kernel(packed_a, packed_b, c_ref, mr, nr, mr, nr, ldc, kc);
    gemm::gotoblas_microkernel_for_shape(
        shape_name,
        packed_a.data(),
        packed_b.data(),
        c_test.data(),
        ldc,
        kc,
        mr,
        nr);

    for (size_t i = 0; i < mr; ++i) {
        for (size_t j = 0; j < nr; ++j) {
            if (!nearly_equal(c_test[i * ldc + j], c_ref[i * ldc + j])) {
                std::cerr << shape_name << " full-tile kernel mismatch at (" << i << ", " << j << ")\n";
                return false;
            }
        }
    }
    return true;
}

bool test_microkernel_shape_fringe_tile(const char* shape_name) {
    if (!gemm::cpu_supports_avx2_fma()) {
        std::cout << "Skipping fixed fringe micro-kernel test (AVX2/FMA unavailable)\n";
        return true;
    }

    size_t mr = 0;
    size_t nr = 0;
    if (!gemm::gotoblas_try_get_kernel_shape(shape_name, &mr, &nr)) {
        std::cerr << "Unknown micro-kernel shape: " << shape_name << "\n";
        return false;
    }
    const size_t rows = mr - 2;
    const size_t cols = nr - 3;
    const size_t kc = 7;
    const size_t ldc = nr + 5;

    std::vector<Scalar> packed_a(kc * mr, 0.0f);
    std::vector<Scalar> packed_b(kc * nr, 0.0f);
    std::vector<Scalar> c_test(mr * ldc, 0.0f);
    std::vector<Scalar> c_ref(mr * ldc, 0.0f);

    for (size_t k = 0; k < kc; ++k) {
        for (size_t i = 0; i < rows; ++i) {
            packed_a[k * mr + i] =
                static_cast<Scalar>(0.2f * static_cast<float>(i + 1) + 0.1f * static_cast<float>(k));
        }
        for (size_t j = 0; j < cols; ++j) {
            packed_b[k * nr + j] =
                static_cast<Scalar>(0.15f * static_cast<float>(j + 1) - 0.05f * static_cast<float>(k));
        }
    }

    reference_kernel(packed_a, packed_b, c_ref, rows, cols, mr, nr, ldc, kc);
    gemm::gotoblas_microkernel_for_shape(
        shape_name,
        packed_a.data(),
        packed_b.data(),
        c_test.data(),
        ldc,
        kc,
        rows,
        cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (!nearly_equal(c_test[i * ldc + j], c_ref[i * ldc + j])) {
                std::cerr << shape_name << " fringe kernel mismatch at (" << i << ", " << j << ")\n";
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    const char* shapes[] = {
        "avx2_8x8",
        "avx2_12x8",
        "avx2_13x8",
        "avx2_4x16",
        "avx2_5x16",
        "avx2_6x16",
    };

    if (!test_pack_a_micro_panel()) {
        return 1;
    }
    if (!test_pack_b_micro_panel()) {
        return 1;
    }

    for (const char* shape : shapes) {
        if (!test_microkernel_shape_full_tile(shape)) {
            return 1;
        }
        if (!test_microkernel_shape_fringe_tile(shape)) {
            return 1;
        }
    }

    std::cout << "All GEMM fixed-kernel tests passed!\n";
    return 0;
}
