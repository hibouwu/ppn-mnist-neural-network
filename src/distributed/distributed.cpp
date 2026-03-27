#include "distributed/distributed.hpp"

#include <limits>
#include <stdexcept>

#ifdef USE_MPI
#include <mpi.h>
#endif

DistributedContext::DistributedContext(int& argc, char**& argv) {
#ifdef USE_MPI
    int initialized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Initialized failed.");
    }

    if (!initialized) {
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init_thread failed.");
        }
        owns_runtime_ = true;
    }

    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank_) != MPI_SUCCESS ||
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_) != MPI_SUCCESS) {
        throw std::runtime_error("MPI communicator setup failed.");
    }
#else
    (void)argc;
    (void)argv;
#endif
}

DistributedContext::~DistributedContext() {
#ifdef USE_MPI
    if (!owns_runtime_) {
        return;
    }

    int finalized = 0;
    if (MPI_Finalized(&finalized) != MPI_SUCCESS) {
        return;
    }
    if (!finalized) {
        MPI_Finalize();
    }
#endif
}

void DistributedContext::allReduceSum(double* data, std::size_t n) const {
    if (data == nullptr || n == 0) {
        return;
    }

#ifdef USE_MPI
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("MPI allReduceSum count exceeds int range.");
    }
    if (MPI_Allreduce(MPI_IN_PLACE, data, static_cast<int>(n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed for double buffer.");
    }
#else
    (void)data;
    (void)n;
#endif
}

void DistributedContext::allReduceMax(double* data, std::size_t n) const {
    if (data == nullptr || n == 0) {
        return;
    }

#ifdef USE_MPI
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("MPI allReduceMax count exceeds int range.");
    }
    if (MPI_Allreduce(MPI_IN_PLACE, data, static_cast<int>(n), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed for max-reduced double buffer.");
    }
#else
    (void)data;
    (void)n;
#endif
}

std::uint64_t DistributedContext::allReduceSumU64(std::uint64_t value) const {
#ifdef USE_MPI
    std::uint64_t result = 0;
    if (MPI_Allreduce(&value, &result, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Allreduce failed for uint64_t value.");
    }
    return result;
#else
    return value;
#endif
}

DistributedRequest DistributedContext::iallReduceSum(double* data, std::size_t n) const {
    DistributedRequest req;
    if (data == nullptr || n == 0 || world_size_ <= 1) {
        return req;
    }

#ifdef USE_MPI
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("MPI iallReduceSum count exceeds int range.");
    }
    if (MPI_Iallreduce(MPI_IN_PLACE,
                       data,
                       static_cast<int>(n),
                       MPI_DOUBLE,
                       MPI_SUM,
                       MPI_COMM_WORLD,
                       &req.request) != MPI_SUCCESS) {
        throw std::runtime_error("MPI_Iallreduce failed for double buffer.");
    }
#else
    (void)data;
    (void)n;
#endif

    return req;
}

void DistributedContext::wait(DistributedRequest& req) const {
#ifdef USE_MPI
    if (req.request != MPI_REQUEST_NULL) {
        if (MPI_Wait(&req.request, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Wait failed.");
        }
    }
#else
    (void)req;
#endif
}

void DistributedContext::waitAll(std::vector<DistributedRequest>& reqs) const {
    for (auto& req : reqs) {
        wait(req);
    }
}

bool DistributedContext::isNullRequest(const DistributedRequest& req) const {
#ifdef USE_MPI
    return req.request == MPI_REQUEST_NULL;
#else
    return req.completed;
#endif
}
