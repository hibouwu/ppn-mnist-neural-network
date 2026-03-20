#pragma once

#include "batch_source.hpp"
#include "dataset.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class TinyImageNetDataset : public Dataset {
public:
    explicit TinyImageNetDataset(const std::string& data_dir);

    void load() override;
    void prepareStreaming();

    std::size_t trainSampleCount() const { return train_samples_.size(); }
    std::size_t testSampleCount() const { return test_samples_.size(); }
    std::shared_ptr<const BatchSource> makeTrainBatchSource(
        const std::vector<std::size_t>& sample_indices) const;
    std::shared_ptr<const BatchSource> makeTestBatchSource(
        const std::vector<std::size_t>& sample_indices) const;

private:
    std::string data_dir_;
    std::string dataset_root_;
    std::vector<std::string> class_ids_;
    std::unordered_map<std::string, std::size_t> class_to_index_;
    std::vector<std::pair<std::string, std::size_t>> train_samples_;
    std::vector<std::pair<std::string, std::size_t>> test_samples_;

    void resolveDatasetRoot();
    void loadClassIndex();
    Matrix loadSplitImages(const std::vector<std::pair<std::string, std::size_t>>& samples) const;
    Matrix loadSplitLabels(const std::vector<std::pair<std::string, std::size_t>>& samples) const;
    std::vector<std::pair<std::string, std::size_t>> collectTrainSamples() const;
    std::vector<std::pair<std::string, std::size_t>> collectValSamples() const;
};
