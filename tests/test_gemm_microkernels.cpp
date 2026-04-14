#include "gemm/matmul_internal.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool nearlyEqual(Scalar a, Scalar b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

bool test_pack_a_micro_panel() {
    const size_t lda = 7;
    const size_t kc = 3;
    const size_t rows = 2;
    const size_t mr = 4;

    std::vector<Scalar> A{
        1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14
    };
    std::vector<Scalar> packed(kc * mr, -1.0f);

    gemm::pack_a_micro_panel(A.data(), packed.data(), lda, kc, rows, mr);

    const std::vector<Scalar> expected{
        1, 8, 0, 0,
        2, 9, 0, 0,
        3, 10, 0, 0
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
        13, 14, 15, 16, 17, 18
    };
    std::vector<Scalar> packed(kc * nr, -1.0f);

    gemm::pack_b_micro_panel(B.data(), packed.data(), ldb, kc, cols, nr);

    const std::vector<Scalar> expected{
        1, 2, 3, 0,
        7, 8, 9, 0,
        13, 14, 15, 0
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
                      size_t ldc,
                      size_t kc) {
    for (size_t i = 0; i < mr; ++i) {
        for (size_t j = 0; j < nr; ++j) {
            Scalar acc = c[i * ldc + j];
            for (size_t k = 0; k < kc; ++k) {
                acc += packed_a[k * mr + i] * packed_b[k * nr + j];
            }
            c[i * ldc + j] = acc;
        }
    }
}

bool test_kernel(const std::string& name) {
    const gemm::KernelSpec* spec = gemm::find_kernel_by_name(name.c_str());
    if (!spec) {
        std::cerr << "Unknown kernel spec: " << name << "\n";
        return false;
    }

    if (spec->isa == gemm::KernelIsa::Avx2 && !gemm::cpu_supports_avx2_fma()) {
        std::cout << "Skipping " << name << " (AVX2/FMA unavailable)\n";
        return true;
    }
    if (spec->isa == gemm::KernelIsa::Avx512 && !gemm::cpu_supports_avx512f()) {
        std::cout << "Skipping " << name << " (AVX-512F unavailable)\n";
        return true;
    }

    const size_t kc = 5;
    const size_t ldc = spec->nr + 3;

    std::vector<Scalar> packed_a(kc * spec->mr);
    std::vector<Scalar> packed_b(kc * spec->nr);
    std::vector<Scalar> c_test(spec->mr * ldc, 0.25f);
    std::vector<Scalar> c_ref(spec->mr * ldc, 0.25f);

    for (size_t k = 0; k < kc; ++k) {
        for (size_t i = 0; i < spec->mr; ++i) {
            packed_a[k * spec->mr + i] =
                static_cast<Scalar>((static_cast<int>(i) - 2) * 0.5f + static_cast<float>(k));
        }
        for (size_t j = 0; j < spec->nr; ++j) {
            packed_b[k * spec->nr + j] =
                static_cast<Scalar>((static_cast<int>(j) - 3) * 0.25f - static_cast<float>(k) * 0.1f);
        }
    }

    reference_kernel(packed_a, packed_b, c_ref, spec->mr, spec->nr, ldc, kc);
    spec->fn(packed_a.data(), packed_b.data(), c_test.data(), ldc, kc);

    for (size_t i = 0; i < spec->mr; ++i) {
        for (size_t j = 0; j < spec->nr; ++j) {
            const Scalar got = c_test[i * ldc + j];
            const Scalar expected = c_ref[i * ldc + j];
            if (!nearlyEqual(got, expected)) {
                std::cerr << "Kernel " << name << " mismatch at (" << i << ", " << j
                          << "): got " << got << ", expected " << expected << "\n";
                return false;
            }
        }
    }

    return true;
}

}  // namespace

int main() {
    if (!test_pack_a_micro_panel()) {
        return 1;
    }
    if (!test_pack_b_micro_panel()) {
        return 1;
    }

    const char* kernels[] = {
        "avx2_6x8",
        "avx2_4x8",
        "avx2_5x8",
        "avx2_8x8",
        "avx2_8x4",
        "avx2_4x16",
        "avx2_6x16",
        "avx512_8x16",
    };

    for (const char* kernel : kernels) {
        if (!test_kernel(kernel)) {
            return 1;
        }
    }

    std::cout << "All GEMM micro-kernel tests passed!\n";
    return 0;
}
