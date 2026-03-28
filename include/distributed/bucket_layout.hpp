#pragma once

#include "distributed/param_registry.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class BucketLayout {
public:
    struct ParamRef {
        Node::Ptr param;
        std::size_t offset_elems = 0;
        std::size_t length_elems = 0;
    };

    struct Bucket {
        std::vector<ParamRef> params;
        std::vector<double> buffer;
    };

    struct DescriptorEntry {
        std::size_t global_param_ordinal = 0;
        std::string logical_param_key;
        std::size_t numel = 0;
        const char* dtype = "double";
        std::size_t bucket_id = 0;
        std::size_t offset = 0;
        std::size_t length = 0;
    };

    BucketLayout(const ParamRegistry& registry, std::size_t bucket_size_bytes);

    std::size_t bucketCount() const { return buckets_.size(); }
    std::optional<std::size_t> bucketIndexFor(const Node& param) const;
    const Bucket& bucket(std::size_t bucket_idx) const { return buckets_.at(bucket_idx); }
    Bucket& bucket(std::size_t bucket_idx) { return buckets_.at(bucket_idx); }
    std::uint64_t bucketBytes(std::size_t bucket_idx) const;
    const std::vector<DescriptorEntry>& descriptorEntries() const { return descriptor_entries_; }
    std::string serializedDescriptor() const;

    void packBucket(std::size_t bucket_idx,
                    const std::unordered_set<const Node*>& touched_params);
    void unpackBucket(std::size_t bucket_idx,
                      const std::unordered_set<const Node*>& touched_params);

private:
    std::vector<Bucket> buckets_;
    std::vector<DescriptorEntry> descriptor_entries_;
    std::unordered_map<const Node*, std::size_t> bucket_index_by_param_;
};
