#pragma once

#include "io/run_artifacts_writer.hpp"
#include "profiling.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace io {

std::string classifyProfileScope(const char* name);
long long opStatsTotalUs(const std::vector<OpTimingStat>& stats);
void accumulateRunOpStats(std::unordered_map<std::string, RunOpStat>& run_stats,
                          const std::vector<OpTimingStat>& op_stats);
std::vector<RunOpStat> makeSortedRunOpStats(
    const std::unordered_map<std::string, RunOpStat>& run_stats);

}
