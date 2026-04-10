#pragma once

#include <cstddef>
#include <vector>

struct MatmulImplStats {
    const char* name = "";
    std::size_t calls = 0;
    long long total_us = 0;
};

struct MatmulEpochStats {
    std::size_t total_calls = 0;
    long long total_us = 0;
    MatmulImplStats per_impl[10]; // blas, ijk, ikj, blocked, omp, omp_blocked, omp_blocked_packb, omp_blocked_packab, omp_gotoblas_avx2, omp_gotoblas_avx512
};

void matmulProfileEpochReset();
void matmulProfileRecord(const char* impl, long long us);
MatmulEpochStats matmulProfileEpochSnapshot();

struct OpTimingStat {
    const char* name = "";
    std::size_t calls = 0;
    long long total_us = 0;
};

void opProfileEpochReset();
void opProfileRecord(const char* name, long long us);
std::vector<OpTimingStat> opProfileEpochSnapshot();

bool vtuneMarkersEnabled();
void vtuneTaskBegin(const char* name);
void vtuneTaskEnd();

class ScopedProfileTask {
public:
    explicit ScopedProfileTask(const char* name);
    ~ScopedProfileTask();

    ScopedProfileTask(const ScopedProfileTask&) = delete;
    ScopedProfileTask& operator=(const ScopedProfileTask&) = delete;

private:
    bool active_ = false;
};
