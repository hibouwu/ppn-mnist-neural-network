#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

struct DistributedRequest {
#ifdef USE_MPI
    MPI_Request request = MPI_REQUEST_NULL;
#else
    bool completed = true;
#endif
};

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
    DistributedRequest iallReduceSum(double* data, std::size_t n) const;
    void wait(DistributedRequest& req) const;
    void waitAll(std::vector<DistributedRequest>& reqs) const;
    bool isNullRequest(const DistributedRequest& req) const;
    std::vector<std::string> allGatherStrings(const std::string& local) const;

private:
    int rank_ = 0;
    int world_size_ = 1;
    bool owns_runtime_ = false;
};
