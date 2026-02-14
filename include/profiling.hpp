#pragma once

#include <cstddef>

struct MatmulImplStats {
    const char* name = "";
    std::size_t calls = 0;
    long long total_us = 0;
};

struct MatmulEpochStats {
    std::size_t total_calls = 0;
    long long total_us = 0;
    MatmulImplStats per_impl[5]; // blas, ijk, ikj, blocked, omp
};

void matmulProfileEpochReset();
void matmulProfileRecord(const char* impl, long long us);
MatmulEpochStats matmulProfileEpochSnapshot();
