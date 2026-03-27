#pragma once

#include "tensor.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

class Node;

constexpr std::uint32_t kInvalidNodeIndex = std::numeric_limits<std::uint32_t>::max();

struct GradientContribution {
    std::uint32_t target_index = kInvalidNodeIndex;
    Matrix grad;
};

class InputIndexView {
public:
    InputIndexView() = default;
    InputIndexView(const std::uint32_t* data, std::size_t size) : data_(data), size_(size) {}

    std::size_t size() const { return size_; }
    const std::uint32_t& operator[](std::size_t i) const {
        if (i >= size_) {
            throw std::out_of_range("InputIndexView: index out of range.");
        }
        return data_[i];
    }

private:
    const std::uint32_t* data_ = nullptr;
    std::size_t size_ = 0;
};

template <std::size_t InlineCapacity>
class InlineContributionList {
public:
    InlineContributionList() = default;

    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    const GradientContribution& operator[](std::size_t i) const {
        if (i >= size_) {
            throw std::out_of_range("InlineContributionList: index out of range.");
        }
        return *storage_[i];
    }

    GradientContribution& operator[](std::size_t i) {
        if (i >= size_) {
            throw std::out_of_range("InlineContributionList: index out of range.");
        }
        return *storage_[i];
    }

    void push_back(GradientContribution contribution) {
        if (size_ >= InlineCapacity) {
            throw std::overflow_error("InlineContributionList: inline capacity exceeded.");
        }
        storage_[size_].emplace(std::move(contribution));
        ++size_;
    }

private:
    std::array<std::optional<GradientContribution>, InlineCapacity> storage_{};
    std::size_t size_ = 0;
};

using ContributionList = InlineContributionList<4>;

class GradFn {
public:
    virtual ~GradFn() = default;
    virtual ContributionList apply(const Node& output,
                                   const Matrix& grad_output,
                                   InputIndexView input_indices) const = 0;
};
