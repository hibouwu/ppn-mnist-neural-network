#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <omp.h>

namespace {

constexpr float kAbsTol = 1e-4f;
constexpr float kRelTol = 1e-5f;

class EnvVarGuard {
public:
    EnvVarGuard(const char* name, const char* value) : name_(name) {
        const char* old_value = std::getenv(name);
        if (old_value != nullptr) {
            had_value_ = true;
            old_value_ = old_value;
        }
        setenv(name, value, 1);
    }

    ~EnvVarGuard() {
        if (had_value_) {
            setenv(name_.c_str(), old_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    EnvVarGuard(const EnvVarGuard&) = delete;
    EnvVarGuard& operator=(const EnvVarGuard&) = delete;

private:
    std::string name_;
    bool had_value_ = false;
    std::string old_value_;
};

bool nearly_equal(Scalar a, Scalar b) {
    const float diff = std::fabs(a - b);
    const float scale = std::max(std::fabs(a), std::fabs(b));
    return diff <= kAbsTol || diff <= kRelTol * scale;
}

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

    float max_abs_error = -1.0f;
    float max_rel_error = -1.0f;
    size_t max_i = 0;
    size_t max_j = 0;
    Scalar max_got = 0.0f;
    Scalar max_expected = 0.0f;
    for (size_t i = 0; i < got.rows; ++i) {
        for (size_t j = 0; j < got.cols; ++j) {
            const Scalar got_value = got(i, j);
            const Scalar expected_value = expected(i, j);
            const float abs_error = std::fabs(got_value - expected_value);
            const float scale = std::max(std::fabs(got_value), std::fabs(expected_value));
            const float rel_error = scale > 0.0f ? abs_error / scale : 0.0f;
            if (abs_error > max_abs_error) {
                max_abs_error = abs_error;
                max_rel_error = rel_error;
                max_i = i;
                max_j = j;
                max_got = got_value;
                max_expected = expected_value;
            }
        }
    }

    if (!nearly_equal(max_got, max_expected)) {
        std::cerr << label << ": max error exceeds tolerance at (" << max_i << ", " << max_j
                  << "), got=" << max_got
                  << ", expected=" << max_expected
                  << ", abs_error=" << max_abs_error
                  << ", rel_error=" << max_rel_error
                  << ", abs_tol=" << kAbsTol
                  << ", rel_tol=" << kRelTol << "\n";
        return false;
    }

    return true;
}

bool run_shape_case(size_t m, size_t n, size_t k, int threads, const std::string& label) {
    Matrix a(m, k, MatrixInit::Uninitialized);
    Matrix b(k, n, MatrixInit::Uninitialized);
    fill_pattern(a, 0.75f);
    fill_pattern(b, -0.5f);

    omp_set_num_threads(threads);

    const Matrix expected = reference_gemm(a, b);
    Matrix got(m, n, MatrixInit::Uninitialized);
    a.matmul_into(b, got);
    return assert_matrix_eq(got, expected, label);
}

bool test_driver_shapes(const char* kernel_name) {
    const struct ShapeCase {
        size_t m;
        size_t n;
        size_t k;
        int threads;
        const char* label;
    } cases[] = {
        {31, 37, 29, 1, "single_thread_square_tail"},
        {61, 11, 27, 2, "small_n_fringe"},
        {13, 97, 19, 4, "wide_n_small_m"},
        {45, 52, 3, 4, "small_k"},
        {35, 41, 22, 4, "tail_heavy"},
        {32, 128, 784, 4, "fc_forward_mainstream_like"},
        {32, 10, 128, 4, "fc_head_small_n_like"},
        {3200, 150, 16, 8, "conv_dx_skinny_k_like"},
    };

    EnvVarGuard kernel_guard("MATMUL_GOTO_KERNEL", kernel_name);
    for (const ShapeCase& c : cases) {
        if (!run_shape_case(c.m, c.n, c.k, c.threads, c.label)) {
            return false;
        }
    }

    return true;
}
}  // namespace

int main() {
    // Fix block sizes before the first matmul call; the runtime accessors cache
    // their first observed values. These values make the baseline driver cases
    // exercise multiple jc panels and M/N/K fringe paths.
    EnvVarGuard mc_guard("MATMUL_MC", "16");
    EnvVarGuard nc_guard("MATMUL_NC", "32");
    EnvVarGuard kc_guard("MATMUL_KC", "20");

    const char* kernels[] = {
        "avx2_8x8",
        "avx2_12x8",
        "avx2_13x8",
        "avx2_4x16",
        "avx2_5x16",
        "avx2_6x16",
    };

    EnvVarGuard impl_guard("MATMUL_IMPL", "omp_gotoblas_avx2");

    for (const char* kernel : kernels) {
        if (!test_driver_shapes(kernel)) {
            return 1;
        }
    }

    std::cout << "GotoBLAS fixed-path driver tests passed!\n";
    return 0;
}
