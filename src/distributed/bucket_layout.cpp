#include "distributed/bucket_layout.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace {

std::size_t checkedBucketSizeElems(std::size_t bucket_size_bytes) {
    if (bucket_size_bytes == 0) {
        throw std::invalid_argument("BucketLayout: bucket_size_bytes must be > 0.");
    }
    const std::size_t elems = bucket_size_bytes / sizeof(Scalar);
    return elems > 0 ? elems : 1;
}

}

BucketLayout::BucketLayout(const ParamRegistry& registry, std::size_t bucket_size_bytes) {
    const std::size_t bucket_size_elems = checkedBucketSizeElems(bucket_size_bytes);

    Bucket current_bucket;
    std::size_t current_elems = 0;
    auto flush_bucket = [&]() {
        if (!current_bucket.params.empty()) {
            current_bucket.buffer.assign(current_elems, 0.0);
            buckets_.push_back(std::move(current_bucket));
            current_bucket = Bucket{};
            current_elems = 0;
        }
    };

    for (const auto& param : registry.params()) {
        const std::size_t param_elems = param->value().data.size();
        if (param_elems == 0) {
            throw std::invalid_argument("BucketLayout: zero-sized parameter is not supported.");
        }

        if (param_elems > bucket_size_elems) {
            flush_bucket();
            Bucket large_bucket;
            large_bucket.params.push_back(ParamRef{param, 0, param_elems});
            large_bucket.buffer.assign(param_elems, 0.0);
            buckets_.push_back(std::move(large_bucket));
            continue;
        }

        if (current_elems + param_elems > bucket_size_elems) {
            flush_bucket();
        }

        current_bucket.params.push_back(ParamRef{param, current_elems, param_elems});
        current_elems += param_elems;
    }
    flush_bucket();

    for (std::size_t bucket_idx = 0; bucket_idx < buckets_.size(); ++bucket_idx) {
        for (const auto& ref : buckets_[bucket_idx].params) {
            const bool inserted = bucket_index_by_param_.emplace(ref.param.get(), bucket_idx).second;
            if (!inserted) {
                throw std::logic_error("BucketLayout: parameter appears in multiple buckets.");
            }
            descriptor_entries_.push_back(DescriptorEntry{
                registry.ordinalFor(*ref.param),
                registry.logicalKeyFor(*ref.param),
                ref.param->value().data.size(),
                "float32",
                bucket_idx,
                ref.offset_elems,
                ref.length_elems});
        }
    }
}

std::optional<std::size_t> BucketLayout::bucketIndexFor(const Node& param) const {
    const auto it = bucket_index_by_param_.find(&param);
    if (it == bucket_index_by_param_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::uint64_t BucketLayout::bucketBytes(std::size_t bucket_idx) const {
    return static_cast<std::uint64_t>(buckets_.at(bucket_idx).buffer.size() * sizeof(Scalar));
}

std::string BucketLayout::serializedDescriptor() const {
    std::ostringstream out;
    out << "bucket_count=" << buckets_.size() << "\n";
    for (const auto& entry : descriptor_entries_) {
        out << entry.global_param_ordinal
            << "|" << entry.logical_param_key
            << "|" << entry.numel
            << "|" << entry.dtype
            << "|" << entry.bucket_id
            << "|" << entry.offset
            << "|" << entry.length
            << "\n";
    }
    return out.str();
}

void BucketLayout::packBucket(std::size_t bucket_idx,
                              const std::unordered_set<const Node*>& touched_params) {
    Bucket& bucket_ref = buckets_.at(bucket_idx);
    std::fill(bucket_ref.buffer.begin(), bucket_ref.buffer.end(), 0.0);

    for (const auto& ref : bucket_ref.params) {
        if (touched_params.count(ref.param.get()) == 0) {
            continue;
        }
        const Matrix& grad = ref.param->grad();
        if (grad.data.size() != ref.length_elems) {
            throw std::logic_error("BucketLayout::packBucket: gradient size does not match parameter size.");
        }
        std::copy(grad.data.begin(),
                  grad.data.end(),
                  bucket_ref.buffer.begin() + static_cast<std::ptrdiff_t>(ref.offset_elems));
    }
}

void BucketLayout::unpackBucket(std::size_t bucket_idx,
                                const std::unordered_set<const Node*>& touched_params) {
    const Bucket& bucket_ref = buckets_.at(bucket_idx);
    for (const auto& ref : bucket_ref.params) {
        if (touched_params.count(ref.param.get()) == 0) {
            continue;
        }
        Matrix& grad = ref.param->grad();
        if (grad.data.size() != ref.length_elems) {
            throw std::logic_error("BucketLayout::unpackBucket: gradient size does not match parameter size.");
        }
        std::copy(bucket_ref.buffer.begin() + static_cast<std::ptrdiff_t>(ref.offset_elems),
                  bucket_ref.buffer.begin() + static_cast<std::ptrdiff_t>(ref.offset_elems + ref.length_elems),
                  grad.data.begin());
    }
}
