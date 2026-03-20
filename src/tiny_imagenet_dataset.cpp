#include "tiny_imagenet_dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <jpeglib.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct DecodedImage {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> pixels;
};

DecodedImage decodeJpegRgb(const std::string& path) {
    FILE* infile = std::fopen(path.c_str(), "rb");
    if (!infile) {
        throw std::runtime_error("Could not open JPEG file: " + path);
    }

    jpeg_decompress_struct cinfo{};
    jpeg_error_mgr jerr{};
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    DecodedImage image;
    image.width = static_cast<int>(cinfo.output_width);
    image.height = static_cast<int>(cinfo.output_height);
    image.channels = static_cast<int>(cinfo.output_components);
    image.pixels.resize(static_cast<std::size_t>(image.width) *
                        static_cast<std::size_t>(image.height) *
                        static_cast<std::size_t>(image.channels));

    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* row_ptr =
            image.pixels.data() +
            static_cast<std::size_t>(cinfo.output_scanline) *
                static_cast<std::size_t>(image.width) *
                static_cast<std::size_t>(image.channels);
        jpeg_read_scanlines(&cinfo, &row_ptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    std::fclose(infile);
    return image;
}

std::vector<std::string> readNonEmptyLines(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

bool hasTinyImageNetLayout(const fs::path& root) {
    return fs::exists(root / "wnids.txt") &&
           fs::exists(root / "train") &&
           fs::exists(root / "val");
}

class TinyImageNetBatchSource : public BatchSource {
public:
    TinyImageNetBatchSource(std::vector<std::pair<std::string, std::size_t>> samples,
                            std::size_t input_dim,
                            std::size_t num_classes)
        : samples_(std::move(samples)),
          input_dim_(input_dim),
          num_classes_(num_classes) {}

    std::size_t rowCount() const override { return samples_.size(); }
    std::size_t inputCols() const override { return input_dim_; }
    std::size_t targetCols() const override { return num_classes_; }

    void loadRows(const std::vector<std::size_t>& indices,
                  Matrix& inputs,
                  Matrix& targets) const override {
        if (inputs.rows != indices.size() || inputs.cols != input_dim_) {
            inputs = Matrix(indices.size(), input_dim_);
        }
        if (targets.rows != indices.size() || targets.cols != num_classes_) {
            targets = Matrix(indices.size(), num_classes_, 0.0);
        } else {
            std::fill(targets.data.begin(), targets.data.end(), 0.0);
        }

        for (std::size_t row = 0; row < indices.size(); ++row) {
            const auto sample_idx = indices[row];
            if (sample_idx >= samples_.size()) {
                throw std::out_of_range("Tiny-ImageNet batch index out of range.");
            }

            const auto& sample = samples_[sample_idx];
            const auto decoded = decodeJpegRgb(sample.first);
            if (decoded.width != 64 || decoded.height != 64 || decoded.channels != 3) {
                throw std::runtime_error(
                    "Tiny-ImageNet image has unexpected shape (expected 64x64x3): " +
                    sample.first);
            }

            double* dst = inputs.data.data() + row * input_dim_;
            for (std::size_t h = 0; h < 64; ++h) {
                for (std::size_t w = 0; w < 64; ++w) {
                    for (std::size_t c = 0; c < 3; ++c) {
                        const std::size_t src_idx = (h * 64 + w) * 3 + c;
                        const std::size_t dst_idx = c * 64 * 64 + h * 64 + w;
                        dst[dst_idx] =
                            static_cast<double>(decoded.pixels[src_idx]) / 255.0;
                    }
                }
            }

            if (sample.second >= num_classes_) {
                throw std::runtime_error("Tiny-ImageNet label out of range.");
            }
            targets(row, sample.second) = 1.0;
        }
    }

private:
    std::vector<std::pair<std::string, std::size_t>> samples_;
    std::size_t input_dim_;
    std::size_t num_classes_;
};

} // namespace

TinyImageNetDataset::TinyImageNetDataset(const std::string& data_dir)
    : data_dir_(data_dir) {
    info_.name = "tiny-imagenet";
    info_.input_channels = 3;
    info_.input_height = 64;
    info_.input_width = 64;
    info_.input_dim = info_.input_channels * info_.input_height * info_.input_width;
    info_.num_classes = 200;
}

void TinyImageNetDataset::resolveDatasetRoot() {
    const fs::path requested_root(data_dir_);
    if (hasTinyImageNetLayout(requested_root)) {
        dataset_root_ = requested_root.string();
        return;
    }

    const fs::path nested_root = requested_root / "tiny-imagenet-200";
    if (hasTinyImageNetLayout(nested_root)) {
        dataset_root_ = nested_root.string();
        return;
    }

    if (!fs::exists(requested_root)) {
        throw std::runtime_error(
            "Tiny-ImageNet dataset directory does not exist: " + requested_root.string() +
            ". Pass the extracted dataset root (the directory containing wnids.txt), "
            "for example --data_dir /abs/path/to/tiny-imagenet-200.");
    }

    throw std::runtime_error(
        "Tiny-ImageNet dataset layout not found under: " + requested_root.string() +
        ". Expected wnids.txt, train/, and val/ either directly in --data_dir or in "
        "--data_dir/tiny-imagenet-200.");
}

void TinyImageNetDataset::loadClassIndex() {
    class_ids_ = readNonEmptyLines(fs::path(dataset_root_) / "wnids.txt");
    if (class_ids_.empty()) {
        throw std::runtime_error("Tiny-ImageNet class index is empty: " +
                                 (fs::path(dataset_root_) / "wnids.txt").string());
    }

    class_to_index_.clear();
    for (std::size_t i = 0; i < class_ids_.size(); ++i) {
        class_to_index_[class_ids_[i]] = i;
    }
    info_.num_classes = class_ids_.size();
}

void TinyImageNetDataset::prepareStreaming() {
    resolveDatasetRoot();
    loadClassIndex();
    train_samples_ = collectTrainSamples();
    test_samples_ = collectValSamples();
}

std::vector<std::pair<std::string, std::size_t>> TinyImageNetDataset::collectTrainSamples() const {
    std::vector<std::pair<std::string, std::size_t>> samples;
    const fs::path train_root = fs::path(dataset_root_) / "train";

    for (const auto& class_id : class_ids_) {
        const auto it = class_to_index_.find(class_id);
        if (it == class_to_index_.end()) {
            throw std::runtime_error("Missing class index for " + class_id);
        }

        const fs::path image_dir = train_root / class_id / "images";
        if (!fs::exists(image_dir)) {
            throw std::runtime_error("Missing Tiny-ImageNet train image directory: " +
                                     image_dir.string());
        }

        std::vector<fs::path> image_paths;
        for (const auto& entry : fs::directory_iterator(image_dir)) {
            if (entry.is_regular_file()) {
                image_paths.push_back(entry.path());
            }
        }
        std::sort(image_paths.begin(), image_paths.end());

        for (const auto& path : image_paths) {
            samples.emplace_back(path.string(), it->second);
        }
    }

    return samples;
}

std::vector<std::pair<std::string, std::size_t>> TinyImageNetDataset::collectValSamples() const {
    const fs::path annotations_path = fs::path(dataset_root_) / "val" / "val_annotations.txt";
    const fs::path image_dir = fs::path(dataset_root_) / "val" / "images";

    std::ifstream in(annotations_path);
    if (!in) {
        throw std::runtime_error("Could not open Tiny-ImageNet val annotations: " +
                                 annotations_path.string());
    }

    std::vector<std::pair<std::string, std::size_t>> samples;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string image_name;
        std::string class_id;
        std::getline(ss, image_name, '\t');
        std::getline(ss, class_id, '\t');
        if (image_name.empty() || class_id.empty()) {
            throw std::runtime_error("Malformed Tiny-ImageNet val annotation line: " + line);
        }

        const auto it = class_to_index_.find(class_id);
        if (it == class_to_index_.end()) {
            throw std::runtime_error("Unknown Tiny-ImageNet class id in val annotations: " + class_id);
        }

        samples.emplace_back((image_dir / image_name).string(), it->second);
    }

    return samples;
}

Matrix TinyImageNetDataset::loadSplitImages(
    const std::vector<std::pair<std::string, std::size_t>>& samples) const {
    Matrix images(samples.size(), info_.input_dim);

    for (std::size_t row = 0; row < samples.size(); ++row) {
        const auto decoded = decodeJpegRgb(samples[row].first);
        if (decoded.width != static_cast<int>(info_.input_width) ||
            decoded.height != static_cast<int>(info_.input_height) ||
            decoded.channels != static_cast<int>(info_.input_channels)) {
            throw std::runtime_error(
                "Tiny-ImageNet image has unexpected shape (expected 64x64x3): " +
                samples[row].first);
        }

        double* dst = images.data.data() + row * info_.input_dim;
        for (std::size_t h = 0; h < info_.input_height; ++h) {
            for (std::size_t w = 0; w < info_.input_width; ++w) {
                for (std::size_t c = 0; c < info_.input_channels; ++c) {
                    const std::size_t src_idx =
                        (h * info_.input_width + w) * info_.input_channels + c;
                    const std::size_t dst_idx =
                        c * info_.input_height * info_.input_width +
                        h * info_.input_width + w;
                    dst[dst_idx] =
                        static_cast<double>(decoded.pixels[src_idx]) / 255.0;
                }
            }
        }
    }

    return images;
}

Matrix TinyImageNetDataset::loadSplitLabels(
    const std::vector<std::pair<std::string, std::size_t>>& samples) const {
    Matrix labels(samples.size(), info_.num_classes, 0.0);
    for (std::size_t row = 0; row < samples.size(); ++row) {
        if (samples[row].second >= info_.num_classes) {
            throw std::runtime_error("Tiny-ImageNet label out of range.");
        }
        labels(row, samples[row].second) = 1.0;
    }
    return labels;
}

void TinyImageNetDataset::load() {
    prepareStreaming();

    train_images_ = loadSplitImages(train_samples_);
    train_labels_ = loadSplitLabels(train_samples_);
    test_images_ = loadSplitImages(test_samples_);
    test_labels_ = loadSplitLabels(test_samples_);
}

std::shared_ptr<const BatchSource> TinyImageNetDataset::makeTrainBatchSource(
    const std::vector<std::size_t>& sample_indices) const {
    if (train_samples_.empty()) {
        throw std::runtime_error("Tiny-ImageNet train split is not prepared.");
    }

    std::vector<std::pair<std::string, std::size_t>> samples;
    samples.reserve(sample_indices.size());
    for (const auto idx : sample_indices) {
        if (idx >= train_samples_.size()) {
            throw std::out_of_range("Tiny-ImageNet train shard index out of range.");
        }
        samples.push_back(train_samples_[idx]);
    }

    return std::make_shared<TinyImageNetBatchSource>(
        std::move(samples), info_.input_dim, info_.num_classes);
}

std::shared_ptr<const BatchSource> TinyImageNetDataset::makeTestBatchSource(
    const std::vector<std::size_t>& sample_indices) const {
    if (test_samples_.empty()) {
        throw std::runtime_error("Tiny-ImageNet test split is not prepared.");
    }

    std::vector<std::pair<std::string, std::size_t>> samples;
    samples.reserve(sample_indices.size());
    for (const auto idx : sample_indices) {
        if (idx >= test_samples_.size()) {
            throw std::out_of_range("Tiny-ImageNet test shard index out of range.");
        }
        samples.push_back(test_samples_[idx]);
    }

    return std::make_shared<TinyImageNetBatchSource>(
        std::move(samples), info_.input_dim, info_.num_classes);
}
