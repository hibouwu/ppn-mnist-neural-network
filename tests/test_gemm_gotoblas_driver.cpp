#include "gemm/matmul_internal.hpp"
#include "tensor.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

namespace {

bool nearly_equal(Scalar a, Scalar b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

constexpr size_t kTestPackM = 16;
constexpr size_t kTestPackN = 24;
constexpr size_t kTestPackK = 20;

void fill_pattern(Matrix& m, float scale) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            const int centered_i = static_cast<int>(i % 11) - 5;
            const int centered_j = static_cast<int>(j % 13) - 6;
            m(i, j) = static_cast<Scalar>(scale * (0.5f * centered_i - 0.25f * centered_j));
        }
    }
}

Matrix reference_gemm(const Matrix& a, const Matrix& b) {
    Matrix out(a.rows, b.cols, MatrixInit::Zero);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t k = 0; k < a.cols; ++k) {
            const Scalar aik = a(i, k);
            for (size_t j = 0; j < b.cols; ++j) {
                out(i, j) += aik * b(k, j);
            }
        }
    }
    return out;
}

bool assert_matrix_eq(const Matrix& got, const Matrix& expected, const std::string& label) {
    if (got.rows != expected.rows || got.cols != expected.cols) {
        std::cerr << label << ": shape mismatch\n";
        return false;
    }

    for (size_t i = 0; i < got.rows; ++i) {
        for (size_t j = 0; j < got.cols; ++j) {
            if (!nearly_equal(got(i, j), expected(i, j))) {
                std::cerr << label << ": mismatch at (" << i << ", " << j
                          << "), got=" << got(i, j)
                          << ", expected=" << expected(i, j) << "\n";
                return false;
            }
        }
    }

    return true;
}

bool run_shape_case(size_t m, size_t n, size_t k, const std::string& label) {
    Matrix a(m, k, MatrixInit::Uninitialized);
    Matrix b(k, n, MatrixInit::Uninitialized);
    fill_pattern(a, 0.75f);
    fill_pattern(b, -0.5f);

    const Matrix expected = reference_gemm(a, b);
    Matrix got(m, n, MatrixInit::Uninitialized);
    a.matmul_into(b, got);
    return assert_matrix_eq(got, expected, label);
}

bool run_shape_case_with_threads(size_t m,
                                 size_t n,
                                 size_t k,
                                 int threads,
                                 const std::string& label,
                                 bool check_pack_count) {
    omp_set_num_threads(threads);
    gemm::reset_gotoblas_debug_counters();

    if (!run_shape_case(m, n, k, label)) {
        return false;
    }

    if (!check_pack_count) {
        return true;
    }

    const size_t expected_pack_calls =
        ((n + kTestPackN - 1) / kTestPackN) * ((k + kTestPackK - 1) / kTestPackK);
    const size_t actual_pack_calls = gemm::gotoblas_pack_bc_call_count();
    if (actual_pack_calls != expected_pack_calls) {
        std::cerr << label << ": pack_Bc call count mismatch, got "
                  << actual_pack_calls << ", expected " << expected_pack_calls << "\n";
        return false;
    }

    return true;
}

bool test_full_driver_shapes() {
    const struct ShapeCase {
        size_t m;
        size_t n;
        size_t k;
        const char* label;
    } cases[] = {
        {31, 37, 29, "square_tail"},
        {61, 11, 27, "skinny_n"},
        {13, 97, 19, "wide_n"},
        {45, 52, 3, "small_k"},
        {35, 41, 22, "tail_heavy"},
    };

    for (const ShapeCase& c : cases) {
        if (!run_shape_case(c.m, c.n, c.k, c.label)) {
            return false;
        }
    }

    return true;
}

bool test_thread_count_coverage() {
    const int thread_counts[] = {1, 2, 4};
    for (int threads : thread_counts) {
        const std::string prefix = "threads_" + std::to_string(threads);
        if (!run_shape_case_with_threads(31, 37, 29, threads, prefix + "_square_tail", true)) {
            return false;
        }
        if (!run_shape_case_with_threads(17, 9, 21, threads, prefix + "_edge_mix", true)) {
            return false;
        }
    }

    return true;
}

bool test_pack_bc_count_independent_of_m() {
    constexpr size_t n = 55;
    constexpr size_t k = 43;
    const size_t m_values[] = {7, 53, 145};

    for (size_t m : m_values) {
        const std::string label = "pack_bc_m_" + std::to_string(m);
        if (!run_shape_case_with_threads(m, n, k, 4, label, true)) {
            return false;
        }
    }

    return true;
}

bool test_strong_edge_case() {
    const gemm::KernelSpec& kernel = gemm::current_kernel_for_isa(gemm::KernelIsa::Avx2);
    const size_t m = kTestPackM + 1;
    const size_t n = kernel.nr + 1;
    const size_t k = kTestPackK + 1;
    return run_shape_case_with_threads(m, n, k, 4, "strong_edge_case", true);
}

bool test_pack_bc_reuse_reference_case() {
    constexpr size_t m = 53;
    constexpr size_t n = 55;
    constexpr size_t k = 43;

    Matrix a(m, k, MatrixInit::Uninitialized);
    Matrix b(k, n, MatrixInit::Uninitialized);
    fill_pattern(a, 1.0f);
    fill_pattern(b, -1.0f);

    omp_set_num_threads(4);
    gemm::reset_gotoblas_debug_counters();
    Matrix got = a.matmul(b);

    const Matrix expected = reference_gemm(a, b);
    if (!assert_matrix_eq(got, expected, "pack_bc_reuse_reference")) {
        return false;
    }

    const size_t expected_pack_calls =
        ((n + kTestPackN - 1) / kTestPackN) * ((k + kTestPackK - 1) / kTestPackK);
    const size_t actual_pack_calls = gemm::gotoblas_pack_bc_call_count();
    if (actual_pack_calls != expected_pack_calls) {
        std::cerr << "pack_Bc call count mismatch: got " << actual_pack_calls
                  << ", expected " << expected_pack_calls << "\n";
        return false;
    }

    return true;
}

}  // namespace

int main() {
    if (!gemm::cpu_supports_avx2_fma()) {
        std::cout << "Skipping GotoBLAS driver test (AVX2/FMA unavailable)\n";
        return 0;
    }

    setenv("MATMUL_IMPL", "omp_gotoblas_avx2", 1);
    setenv("MATMUL_PACK_M", "16", 1);
    setenv("MATMUL_PACK_N", "24", 1);
    setenv("MATMUL_PACK_K", "20", 1);
    omp_set_num_threads(4);

    if (!test_full_driver_shapes()) {
        return 1;
    }
    if (!test_thread_count_coverage()) {
        return 1;
    }
    if (!test_pack_bc_count_independent_of_m()) {
        return 1;
    }
    if (!test_strong_edge_case()) {
        return 1;
    }
    if (!test_pack_bc_reuse_reference_case()) {
        return 1;
    }

    std::cout << "GotoBLAS driver tests passed!\n";
    return 0;
}
