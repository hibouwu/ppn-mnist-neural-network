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
    MatmulImplStats per_impl[5]; // blas, ijk, ikj, blocked, omp
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
