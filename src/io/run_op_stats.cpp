#include "io/run_op_stats.hpp"

#include <algorithm>
#include <string_view>

namespace io {

std::string classifyProfileScope(const char* name) {
    const std::string_view n(name != nullptr ? name : "");
    if (n.rfind("engine_", 0) == 0) {
        return "engine";
    }
    if (n.rfind("sync_", 0) == 0) {
        return "sync";
    }
    if (n.rfind("optimizer_", 0) == 0) {
        return "optimizer";
    }
    return "op";
}

long long opStatsTotalUs(const std::vector<OpTimingStat>& stats) {
    long long total = 0;
    for (const auto& stat : stats) {
        total += stat.total_us;
    }
    return total;
}

void accumulateRunOpStats(std::unordered_map<std::string, RunOpStat>& run_stats,
                          const std::vector<OpTimingStat>& op_stats) {
    for (const auto& stat : op_stats) {
        const std::string scope = classifyProfileScope(stat.name);
        const std::string key = scope + "|" + stat.name;
        auto& entry = run_stats[key];
        if (entry.name.empty()) {
            entry.scope = scope;
            entry.name = stat.name;
        }
        entry.calls += stat.calls;
        entry.total_us += stat.total_us;
    }
}

std::vector<RunOpStat> makeSortedRunOpStats(
    const std::unordered_map<std::string, RunOpStat>& run_stats) {
    std::vector<RunOpStat> rows;
    rows.reserve(run_stats.size());
    for (const auto& [_, stat] : run_stats) {
        rows.push_back(stat);
    }
    std::sort(rows.begin(), rows.end(), [](const RunOpStat& a, const RunOpStat& b) {
        return a.total_us > b.total_us;
    });
    return rows;
}

}
