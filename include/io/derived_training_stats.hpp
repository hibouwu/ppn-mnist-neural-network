#pragma once

#include "trainer.hpp"

namespace io {

struct DerivedTrainingStats {
    double avg_step_time_ms = 0.0;
    double max_step_time_ms = 0.0;
    double samples_per_s = 0.0;
    double allreduce_wait_ratio = 0.0;
};

DerivedTrainingStats deriveTrainingStats(const Metrics& train_metrics);

}
