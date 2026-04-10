#include "profiling.hpp"
#include <cstring>
#include <string>
#include <unordered_map>

#if defined(PPN_HAVE_VTUNE) && PPN_HAVE_VTUNE
#include <ittnotify.h>
#endif

namespace {
MatmulEpochStats g_stats;
std::unordered_map<std::string, OpTimingStat> g_op_stats;

int implIndex(const char* impl) {
    if (std::strcmp(impl, "blas") == 0) return 0;
    if (std::strcmp(impl, "ijk") == 0) return 1;
    if (std::strcmp(impl, "ikj") == 0) return 2;
    if (std::strcmp(impl, "blocked") == 0) return 3;
    if (std::strcmp(impl, "omp") == 0) return 4;
    if (std::strcmp(impl, "omp_blocked") == 0) return 5;
    if (std::strcmp(impl, "omp_blocked_packb") == 0) return 6;
    if (std::strcmp(impl, "omp_blocked_packab") == 0) return 7;
    if (std::strcmp(impl, "omp_gotoblas_avx2") == 0) return 8;
    if (std::strcmp(impl, "omp_gotoblas_avx512") == 0) return 9;
    return -1;
}

void initNames() {
    g_stats.per_impl[0].name = "blas";
    g_stats.per_impl[1].name = "ijk";
    g_stats.per_impl[2].name = "ikj";
    g_stats.per_impl[3].name = "blocked";
    g_stats.per_impl[4].name = "omp";
    g_stats.per_impl[5].name = "omp_blocked";
    g_stats.per_impl[6].name = "omp_blocked_packb";
    g_stats.per_impl[7].name = "omp_blocked_packab";
    g_stats.per_impl[8].name = "omp_gotoblas_avx2";
    g_stats.per_impl[9].name = "omp_gotoblas_avx512";
}

#if defined(PPN_HAVE_VTUNE) && PPN_HAVE_VTUNE
__itt_domain* vtuneDomain() {
    static __itt_domain* domain = __itt_domain_create("ppn");
    return domain;
}
#endif
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

void opProfileEpochReset() {
    g_op_stats.clear();
}

void opProfileRecord(const char* name, long long us) {
    auto [it, inserted] = g_op_stats.emplace(std::string(name), OpTimingStat{name, 0, 0});
    it->second.calls += 1;
    it->second.total_us += us;
    if (inserted) {
        it->second.name = it->first.c_str();
    }
}

std::vector<OpTimingStat> opProfileEpochSnapshot() {
    std::vector<OpTimingStat> out;
    out.reserve(g_op_stats.size());
    for (const auto& [_, stat] : g_op_stats) {
        out.push_back(stat);
    }
    return out;
}

bool vtuneMarkersEnabled() {
#if defined(PPN_HAVE_VTUNE) && PPN_HAVE_VTUNE
    return true;
#else
    return false;
#endif
}

void vtuneTaskBegin(const char* name) {
#if defined(PPN_HAVE_VTUNE) && PPN_HAVE_VTUNE
    __itt_task_begin(vtuneDomain(),
                     __itt_null,
                     __itt_null,
                     __itt_string_handle_create(name));
#else
    (void)name;
#endif
}

void vtuneTaskEnd() {
#if defined(PPN_HAVE_VTUNE) && PPN_HAVE_VTUNE
    __itt_task_end(vtuneDomain());
#endif
}

ScopedProfileTask::ScopedProfileTask(const char* name)
    : active_(vtuneMarkersEnabled()) {
    if (active_) {
        vtuneTaskBegin(name);
    }
}

ScopedProfileTask::~ScopedProfileTask() {
    if (active_) {
        vtuneTaskEnd();
    }
}
