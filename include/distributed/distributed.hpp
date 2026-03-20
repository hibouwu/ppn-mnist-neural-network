#pragma once

#include <cstddef>
#include <cstdint>

class DistributedContext {
public:
    DistributedContext(int& argc, char**& argv);
    ~DistributedContext();

    DistributedContext(const DistributedContext&) = delete;
    DistributedContext& operator=(const DistributedContext&) = delete;

    int rank() const { return rank_; }
    int worldSize() const { return world_size_; }
    bool isMaster() const { return rank_ == 0; }

    void allReduceSum(double* data, std::size_t n) const;
    void allReduceMax(double* data, std::size_t n) const;
    std::uint64_t allReduceSumU64(std::uint64_t value) const;

private:
    int rank_ = 0;
    int world_size_ = 1;
    bool owns_runtime_ = false;
};
