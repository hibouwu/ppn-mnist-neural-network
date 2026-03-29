#include "distributed/param_registry.hpp"
#include "synchronizable_param.hpp"

#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace {

std::uint64_t fnv1aAppend(std::uint64_t hash, std::uint64_t value) {
    constexpr std::uint64_t kOffset = 1469598103934665603ULL;
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    if (hash == 0) {
        hash = kOffset;
    }
    for (std::size_t shift = 0; shift < sizeof(value) * 8; shift += 8) {
        const auto byte = static_cast<unsigned char>((value >> shift) & 0xffU);
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= kPrime;
    }
    return hash;
}

std::string buildLogicalKey(const Node& param, std::size_t ordinal) {
    const Matrix& value = param.value();
    std::uint64_t hash = 0;
    hash = fnv1aAppend(hash, static_cast<std::uint64_t>(ordinal));
    hash = fnv1aAppend(hash, static_cast<std::uint64_t>(value.rows));
    hash = fnv1aAppend(hash, static_cast<std::uint64_t>(value.cols));
    hash = fnv1aAppend(hash, static_cast<std::uint64_t>(value.data.size()));
    for (Scalar v : value.data) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(v));
        hash = fnv1aAppend(hash, bits);
    }

    std::ostringstream out;
    out << "ordinal=" << ordinal
        << ";shape=" << value.rows << "x" << value.cols
        << ";numel=" << value.data.size()
        << ";dtype=float32"
        << ";value_hash=" << hash;
    return out.str();
}

}

ParamRegistry::ParamRegistry(const std::vector<Node::Ptr>& params) {
    params_.reserve(params.size());
    logical_keys_.reserve(params.size());
    std::unordered_set<const Node*> seen;
    for (const auto& param : params) {
        if (!param) {
            throw std::invalid_argument("ParamRegistry: null parameter in parameter list.");
        }
        if (!isSynchronizableLeafParameter(*param)) {
            throw std::invalid_argument(
                "ParamRegistry: expected synchronizable leaf parameters only.");
        }
        const bool inserted = seen.insert(param.get()).second;
        if (!inserted) {
            throw std::logic_error("ParamRegistry: duplicate parameter identity detected.");
        }
        params_.push_back(param);
        logical_keys_.push_back(buildLogicalKey(*param, params_.size() - 1));
    }
}

bool ParamRegistry::contains(const Node& param) const {
    for (std::size_t idx = 0; idx < params_.size(); ++idx) {
        if (params_[idx].get() == &param) {
            return true;
        }
    }
    return false;
}

std::size_t ParamRegistry::ordinalFor(const Node& param) const {
    for (std::size_t idx = 0; idx < params_.size(); ++idx) {
        if (params_[idx].get() == &param) {
            return idx;
        }
    }
    throw std::logic_error("ParamRegistry::ordinalFor: parameter missing from registry.");
}

const std::string& ParamRegistry::logicalKeyFor(const Node& param) const {
    return logical_keys_.at(ordinalFor(param));
}
