#include "profiling.hpp"
#include <cstring>

namespace {
MatmulEpochStats g_stats;

int implIndex(const char* impl) {
    if (std::strcmp(impl, "blas") == 0) return 0;
    if (std::strcmp(impl, "ijk") == 0) return 1;
    if (std::strcmp(impl, "ikj") == 0) return 2;
    if (std::strcmp(impl, "blocked") == 0) return 3;
    if (std::strcmp(impl, "omp") == 0) return 4;
    return -1;
}

void initNames() {
    g_stats.per_impl[0].name = "blas";
    g_stats.per_impl[1].name = "ijk";
    g_stats.per_impl[2].name = "ikj";
    g_stats.per_impl[3].name = "blocked";
    g_stats.per_impl[4].name = "omp";
}
}

void matmulProfileEpochReset() {
    g_stats = MatmulEpochStats{};
    initNames();
}

void matmulProfileRecord(const char* impl, long long us) {
    const int idx = implIndex(impl);
    g_stats.total_calls += 1;
    g_stats.total_us += us;

    if (idx >= 0) {
        g_stats.per_impl[idx].calls += 1;
        g_stats.per_impl[idx].total_us += us;
    }
}

MatmulEpochStats matmulProfileEpochSnapshot() {
    return g_stats;
}
