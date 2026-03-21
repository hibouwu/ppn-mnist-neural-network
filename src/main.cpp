#include <algorithm>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <limits>
#include <cmath>
#include <cstdint>

namespace fs = std::filesystem;

#include "dataset.hpp"
#include "mnist_dataset.hpp"
#include "tiny_imagenet_dataset.hpp"
#include "dataloader.hpp"
#include "distributed/distributed.hpp"
#include "network.hpp"
#include "cnn_network.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "trainer.hpp"
#include "activation.hpp"

#include "profiling.hpp"

// ------------------ helpers ------------------
static long long parseIntStrict(const std::string& s, const std::string& name) {
    if (s.empty()) {
        throw std::invalid_argument(name + ": value must not be empty.");
    }

    size_t pos = 0;
    long long value = 0;
    try {
        value = std::stoll(s, &pos);
    } catch (const std::exception&) {
        throw std::invalid_argument(name + ": invalid integer '" + s + "'.");
    }
    if (pos != s.size()) {
        throw std::invalid_argument(name + ": invalid integer '" + s + "'.");
    }
    return value;
}

static int parseIntInRange(const std::string& s, const std::string& name,
                           int min_value, int max_value) {
    long long value = parseIntStrict(s, name);
    if (value < static_cast<long long>(min_value) ||
        value > static_cast<long long>(max_value)) {
        throw std::invalid_argument(
            name + ": expected value in range [" + std::to_string(min_value) +
            ", " + std::to_string(max_value) + "], got " + s + ".");
    }
    return static_cast<int>(value);
}

static unsigned int parseUnsignedIntStrict(const std::string& s, const std::string& name) {
    long long value = parseIntStrict(s, name);
    if (value < 0 || value > static_cast<long long>(std::numeric_limits<unsigned int>::max())) {
        throw std::invalid_argument(
            name + ": expected value in range [0, " +
            std::to_string(std::numeric_limits<unsigned int>::max()) + "], got " + s + ".");
    }
    return static_cast<unsigned int>(value);
}

static double parseDoubleStrict(const std::string& s, const std::string& name) {
    if (s.empty()) {
        throw std::invalid_argument(name + ": value must not be empty.");
    }

    size_t pos = 0;
    double value = 0.0;
    try {
        value = std::stod(s, &pos);
    } catch (const std::exception&) {
        throw std::invalid_argument(name + ": invalid number '" + s + "'.");
    }
    if (pos != s.size()) {
        throw std::invalid_argument(name + ": invalid number '" + s + "'.");
    }
    if (!std::isfinite(value)) {
        throw std::invalid_argument(name + ": value must be finite.");
    }
    return value;
}

static std::vector<int> parseHiddenSizes(const std::string& s) {
    std::vector<int> hs;
    std::stringstream ss(s);
    std::string item;
    size_t idx = 0;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            throw std::invalid_argument(
                "hidden_sizes: empty value in list (consecutive commas?): '" + s + "'.");
        }
        long long value = parseIntStrict(item, "hidden_sizes[" + std::to_string(idx) + "]");
        if (value <= 0 || value > static_cast<long long>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                "hidden_sizes[" + std::to_string(idx) + "]: expected > 0.");
        }
        hs.push_back(static_cast<int>(value));
        ++idx;
    }
    if (hs.empty()) {
        throw std::invalid_argument("hidden_sizes must contain at least one value.");
    }
    return hs;
}

static std::vector<size_t> parseUintList(const std::string& s, const std::string& name) {
    std::vector<size_t> v;
    std::stringstream ss(s);
    std::string item;
    size_t idx = 0;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            throw std::invalid_argument(
                name + ": empty value in list (consecutive commas?): '" + s + "'.");
        }
        long long val = parseIntStrict(item, name + "[" + std::to_string(idx) + "]");
        if (val < 0) {
            throw std::invalid_argument(
                name + ": negative value not allowed: " + item + ".");
        }
        v.push_back(static_cast<size_t>(val));
        ++idx;
    }
    return v;
}

static std::vector<bool> parseBoolList(const std::string& s, const std::string& name) {
    std::vector<bool> v;
    std::stringstream ss(s);
    std::string item;
    size_t idx = 0;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            throw std::invalid_argument(
                name + ": empty value in list (consecutive commas?): '" + s + "'.");
        }
        if (item == "0") {
            v.push_back(false);
        } else if (item == "1") {
            v.push_back(true);
        }
        else {
            throw std::invalid_argument(
                name + ": expected 0 or 1 at index " + std::to_string(idx) +
                ", got '" + item + "'.");
        }
        ++idx;
    }
    return v;
}

static std::string joinHiddenSizes(const std::vector<int>& hs) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < hs.size(); ++i) {
        os << hs[i];
        if (i + 1 < hs.size()) os << ",";
    }
    os << "]";
    return os.str();
}

static DatasetInfo inferDatasetInfo(const std::string& dataset_name) {
    if (dataset_name == "mnist") {
        DatasetInfo info;
        info.name = "mnist";
        info.input_channels = 1;
        info.input_height = 28;
        info.input_width = 28;
        info.input_dim = 28 * 28;
        info.num_classes = 10;
        return info;
    }
    if (dataset_name == "tiny-imagenet") {
        DatasetInfo info;
        info.name = "tiny-imagenet";
        info.input_channels = 3;
        info.input_height = 64;
        info.input_width = 64;
        info.input_dim = 3 * 64 * 64;
        info.num_classes = 200;
        return info;
    }
    throw std::invalid_argument(
        "unsupported --dataset '" + dataset_name +
        "'. Expected 'mnist' or 'tiny-imagenet'.");
}

static std::unique_ptr<Dataset> createDataset(const std::string& dataset_name,
                                              const std::string& data_dir) {
    if (dataset_name == "mnist") {
        return std::make_unique<MNISTDataset>(data_dir);
    }
    if (dataset_name == "tiny-imagenet") {
        return std::make_unique<TinyImageNetDataset>(data_dir);
    }
    throw std::invalid_argument(
        "unsupported --dataset '" + dataset_name +
        "'. Expected 'mnist' or 'tiny-imagenet'.");
}

struct DatasetShard {
    Matrix inputs;
    Matrix targets;
};

static std::vector<std::size_t> makeRoundRobinIndices(std::size_t total_rows,
                                                      int rank,
                                                      int world_size) {
    std::vector<std::size_t> indices;
    for (std::size_t idx = static_cast<std::size_t>(rank);
         idx < total_rows;
         idx += static_cast<std::size_t>(world_size)) {
        indices.push_back(idx);
    }
    return indices;
}

static std::vector<std::size_t> makePaddedRoundRobinIndices(std::size_t total_rows,
                                                            int rank,
                                                            int world_size) {
    std::vector<std::size_t> indices;
    if (total_rows == 0) {
        return indices;
    }

    const std::size_t target_count =
        (total_rows + static_cast<std::size_t>(world_size) - 1) /
        static_cast<std::size_t>(world_size);
    indices.reserve(target_count);

    for (std::size_t step = 0; step < target_count; ++step) {
        const std::size_t linear_idx =
            static_cast<std::size_t>(rank) + step * static_cast<std::size_t>(world_size);
        indices.push_back((linear_idx < total_rows) ? linear_idx : (linear_idx % total_rows));
    }

    return indices;
}

static DatasetShard makeShard(const Matrix& inputs,
                              const Matrix& targets,
                              const std::vector<std::size_t>& indices) {
    if (inputs.rows != targets.rows) {
        throw std::invalid_argument("makeShard: inputs/targets rows mismatch.");
    }

    DatasetShard shard{
        Matrix(indices.size(), inputs.cols),
        Matrix(indices.size(), targets.cols)
    };

    for (std::size_t row = 0; row < indices.size(); ++row) {
        const std::size_t src_idx = indices[row];
        if (src_idx >= inputs.rows) {
            throw std::out_of_range("makeShard: index out of range.");
        }

        const double* src_x = inputs.data.data() + src_idx * inputs.cols;
        double* dst_x = shard.inputs.data.data() + row * inputs.cols;
        std::copy_n(src_x, inputs.cols, dst_x);

        const double* src_y = targets.data.data() + src_idx * targets.cols;
        double* dst_y = shard.targets.data.data() + row * targets.cols;
        std::copy_n(src_y, targets.cols, dst_y);
    }

    return shard;
}

static Metrics reduceMetrics(const DistributedContext& dist, Metrics metrics) {
    dist.allReduceSum(&metrics.loss_sum, 1);
    metrics.sample_count = dist.allReduceSumU64(metrics.sample_count);
    metrics.correct_count = dist.allReduceSumU64(metrics.correct_count);

    if (metrics.sample_count > 0) {
        metrics.avg_loss = metrics.loss_sum / static_cast<double>(metrics.sample_count);
        metrics.accuracy = static_cast<double>(metrics.correct_count) /
                           static_cast<double>(metrics.sample_count);
    } else {
        metrics.avg_loss = 0.0;
        metrics.accuracy = 0.0;
    }

    return metrics;
}

static EpochProfile reduceEpochProfile(const DistributedContext& dist, EpochProfile profile) {
    double max_fields[] = {
        profile.epoch_time_s,
        profile.data_time_s,
        profile.fwd_bwd_time_s,
        profile.sync_total_time_s,
        profile.sync_wait_time_s,
        profile.opt_time_s,
        profile.step_time_s_sum,
        profile.max_step_time_s
    };
    dist.allReduceMax(max_fields, sizeof(max_fields) / sizeof(max_fields[0]));

    profile.epoch_time_s = max_fields[0];
    profile.data_time_s = max_fields[1];
    profile.fwd_bwd_time_s = max_fields[2];
    profile.sync_total_time_s = max_fields[3];
    profile.sync_wait_time_s = max_fields[4];
    profile.opt_time_s = max_fields[5];
    profile.step_time_s_sum = max_fields[6];
    profile.max_step_time_s = max_fields[7];
    profile.step_count = dist.allReduceSumU64(profile.step_count);
    return profile;
}
// --------------------------------------------

// configuration struct
struct Config {
    int epochs = 1;                       // Number of training epochs. 0 means "build/eval only, no train loop".
    int batch_size = 64;                  // Mini-batch size used by both train/test DataLoader.
    double learning_rate = 0.01;          // Base learning rate (> 0).

    // Backward compatible:
    int hidden_size = 128;                // Legacy single hidden layer width for MLP.
    std::string hidden_sizes = "";        // Preferred MLP hidden sizes, e.g. "256,128,64". Empty -> use hidden_size.

    std::string dataset = "mnist";        // Dataset: mnist / tiny-imagenet.
    std::string data_dir = "mnist";       // MNIST dataset folder path.
    unsigned int seed = 0;                // RNG seed. 0 means "non-deterministic/random seed path".
    std::string activation = "relu";      // MLP activation: relu / leaky_relu / gelu / sigmoid / tanh.
    std::string init = "he";              // MLP init strategy: he / xavier / manual.
    std::string model = "mlp";            // Model type: mlp / cnn.
    std::string optimizer = "momentum_sgd";  // Optimizer: sgd / momentum_sgd(/momentum) / adamw.
    double momentum = 0.9;                // Momentum for momentum_sgd.
    bool nesterov = false;                // Nesterov flag for momentum_sgd.
    double weight_decay = 0.0;            // Weight decay (>= 0).
    double beta1 = 0.9;                   // AdamW beta1 in [0,1).
    double beta2 = 0.999;                 // AdamW beta2 in [0,1).
    double eps = 1e-8;                    // AdamW epsilon (> 0).

    // CNN-specific (only used when model == "cnn")
    std::string cnn_conv_channels = "";   // Required for custom CNN. E.g. "6,16,32". All CNN fields empty -> use LeNet-5 preset.
    std::string cnn_conv_kernels = "";    // Optional. E.g. "5,3,3". Empty -> stage defaults in CNNConfig::expandDefaults.
    std::string cnn_conv_strides = "";    // Optional. E.g. "1,1,1". Empty -> stage defaults.
    std::string cnn_conv_paddings = "";   // Optional. E.g. "2,1,1". Empty -> stage defaults.
    std::string cnn_pool_after = "";      // Optional bool list (0/1), e.g. "1,1,0". Empty -> stage defaults.
    std::string cnn_pool_kernels = "";    // Optional. E.g. "2,2,2". Empty -> stage defaults.
    std::string cnn_pool_strides = "";    // Optional. E.g. "2,2,2". Empty -> stage defaults.
    std::string cnn_fc_hidden_sizes = ""; // Optional FC hidden dims after conv stack. Empty -> default FC layout.

    std::string out_dir = "output";       // Output directory for metrics/log artifacts.
};

int main(int argc, char** argv) {
    DistributedContext dist(argc, argv);
    const bool isMaster = dist.isMaster();
    Config cfg;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto requireValue = [&](const std::string& flag) -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(flag + ": missing value.");
                }
                return argv[++i];
            };

            if (arg == "--epochs") {
                cfg.epochs = parseIntInRange(
                    requireValue(arg), arg, 0, std::numeric_limits<int>::max());
            } else if (arg == "--batch_size") {
                cfg.batch_size = parseIntInRange(
                    requireValue(arg), arg, 1, std::numeric_limits<int>::max());
            } else if (arg == "--learning_rate") {
                cfg.learning_rate = parseDoubleStrict(requireValue(arg), arg);
                if (cfg.learning_rate <= 0.0) {
                    throw std::invalid_argument(arg + ": expected > 0.");
                }
            } else if (arg == "--hidden_size") {
                cfg.hidden_size = parseIntInRange(
                    requireValue(arg), arg, 1, std::numeric_limits<int>::max());
            } else if (arg == "--hidden_sizes") {
                cfg.hidden_sizes = requireValue(arg);
            } else if (arg == "--data_dir") {
                cfg.data_dir = requireValue(arg);
            } else if (arg == "--dataset") {
                cfg.dataset = requireValue(arg);
            } else if (arg == "--seed") {
                cfg.seed = parseUnsignedIntStrict(requireValue(arg), arg);
            } else if (arg == "--activation") {
                cfg.activation = requireValue(arg);
            } else if (arg == "--optimizer") {
                cfg.optimizer = requireValue(arg);
            } else if (arg == "--momentum") {
                cfg.momentum = parseDoubleStrict(requireValue(arg), arg);
                if (cfg.momentum < 0.0 || cfg.momentum >= 1.0) {
                    throw std::invalid_argument(arg + ": expected value in [0, 1).");
                }
            } else if (arg == "--nesterov") {
                cfg.nesterov = (parseIntInRange(requireValue(arg), arg, 0, 1) != 0);
            } else if (arg == "--weight_decay") {
                cfg.weight_decay = parseDoubleStrict(requireValue(arg), arg);
                if (cfg.weight_decay < 0.0) {
                    throw std::invalid_argument(arg + ": expected >= 0.");
                }
            } else if (arg == "--beta1") {
                cfg.beta1 = parseDoubleStrict(requireValue(arg), arg);
                if (cfg.beta1 < 0.0 || cfg.beta1 >= 1.0) {
                    throw std::invalid_argument(arg + ": expected value in [0, 1).");
                }
            } else if (arg == "--beta2") {
                cfg.beta2 = parseDoubleStrict(requireValue(arg), arg);
                if (cfg.beta2 < 0.0 || cfg.beta2 >= 1.0) {
                    throw std::invalid_argument(arg + ": expected value in [0, 1).");
                }
            } else if (arg == "--eps") {
                cfg.eps = parseDoubleStrict(requireValue(arg), arg);
                if (cfg.eps <= 0.0) {
                    throw std::invalid_argument(arg + ": expected > 0.");
                }
            } else if (arg == "--init") {
                cfg.init = requireValue(arg);
            } else if (arg == "--model") {
                cfg.model = requireValue(arg);
            } else if (arg == "--cnn_conv_channels") {
                cfg.cnn_conv_channels = requireValue(arg);
            } else if (arg == "--cnn_conv_kernels") {
                cfg.cnn_conv_kernels = requireValue(arg);
            } else if (arg == "--cnn_conv_strides") {
                cfg.cnn_conv_strides = requireValue(arg);
            } else if (arg == "--cnn_conv_paddings") {
                cfg.cnn_conv_paddings = requireValue(arg);
            } else if (arg == "--cnn_pool_after") {
                cfg.cnn_pool_after = requireValue(arg);
            } else if (arg == "--cnn_pool_kernels") {
                cfg.cnn_pool_kernels = requireValue(arg);
            } else if (arg == "--cnn_pool_strides") {
                cfg.cnn_pool_strides = requireValue(arg);
            } else if (arg == "--cnn_fc_hidden_sizes") {
                cfg.cnn_fc_hidden_sizes = requireValue(arg);
            } else if (arg == "--out_dir") {
                cfg.out_dir = requireValue(arg);
            } else {
                throw std::invalid_argument("unknown argument '" + arg + "'");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 1;
    }

    // Decide architecture
    std::vector<int> hiddenVec;
    try {
        if (!cfg.hidden_sizes.empty()) {
            hiddenVec = parseHiddenSizes(cfg.hidden_sizes);
        } else {
            hiddenVec = { cfg.hidden_size };
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing hidden sizes: " << e.what() << std::endl;
        return 1;
    }

    if (dist.worldSize() > 1 && cfg.seed == 0) {
        if (isMaster) {
            std::cerr << "Error: --seed must be non-zero when running with MPI world size > 1."
                      << std::endl;
        }
        return 1;
    }

    if (isMaster) {
        DatasetInfo datasetInfo;
        try {
            datasetInfo = inferDatasetInfo(cfg.dataset);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        std::cout << "Starting training with config:" << "\n"
                  << "  Dataset: " << cfg.dataset << "\n"
                  << "  Data Dir: " << cfg.data_dir << "\n"
                  << "  Input Shape: " << datasetInfo.input_channels << "x"
                  << datasetInfo.input_height << "x" << datasetInfo.input_width << "\n"
                  << "  Classes: " << datasetInfo.num_classes << "\n"
                  << "  Epochs: " << cfg.epochs << "\n"
                  << "  Batch Size: " << cfg.batch_size << "\n"
                  << "  Learning Rate: " << cfg.learning_rate << "\n"
                  << "  Hidden (sizes): " << joinHiddenSizes(hiddenVec) << "\n"
                  << "  Seed: " << cfg.seed << "\n"
                  << "  Activation: " << cfg.activation << "\n"
                  << "  Init: " << cfg.init << "\n"
                  << "  Model: " << cfg.model << "\n"
                  << "  Optimizer: " << cfg.optimizer << "\n"
                  << "  Momentum: " << cfg.momentum << "\n"
                  << "  Nesterov: " << (cfg.nesterov ? 1 : 0) << "\n"
                  << "  Weight Decay: " << cfg.weight_decay << "\n"
                  << "  Beta1: " << cfg.beta1 << "\n"
                  << "  Beta2: " << cfg.beta2 << "\n"
                  << "  Eps: " << cfg.eps << "\n";
        if (dist.worldSize() > 1) {
            std::cout << "  MPI World Size: " << dist.worldSize() << "\n";
        }
        std::cout << std::endl;
    }

    const bool optimizerIsSGD = (cfg.optimizer == "sgd");
    const bool optimizerIsMomentum = (cfg.optimizer == "momentum_sgd" || cfg.optimizer == "momentum");
    const bool optimizerIsAdamW = (cfg.optimizer == "adamw");
    if (!(optimizerIsSGD || optimizerIsMomentum || optimizerIsAdamW)) {
        std::cerr << "Error: unsupported --optimizer '" << cfg.optimizer
                  << "'. Expected 'sgd', 'momentum_sgd' (or 'momentum'), or 'adamw'."
                  << std::endl;
        return 1;
    }

    DatasetInfo datasetInfo;
    try {
        datasetInfo = inferDatasetInfo(cfg.dataset);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // 1. Build Model (fail-fast: validate config before loading data)
    std::unique_ptr<NeuralNetwork> model;
    if (cfg.model == "cnn") {
        bool hasCustom = !cfg.cnn_conv_channels.empty() ||
                         !cfg.cnn_conv_kernels.empty()  ||
                         !cfg.cnn_conv_strides.empty()  ||
                         !cfg.cnn_conv_paddings.empty() ||
                         !cfg.cnn_pool_after.empty()    ||
                         !cfg.cnn_pool_kernels.empty()  ||
                         !cfg.cnn_pool_strides.empty()  ||
                         !cfg.cnn_fc_hidden_sizes.empty();

        CNNConfig cnnCfg;
        if (hasCustom) {
            if (cfg.cnn_conv_channels.empty()) {
                std::cerr << "Error: --cnn_conv_channels is required when using custom CNN." << std::endl;
                return 1;
            }
            try {
                cnnCfg.conv_channels = parseUintList(cfg.cnn_conv_channels, "cnn_conv_channels");
                if (!cfg.cnn_conv_kernels.empty()) {
                    cnnCfg.conv_kernels = parseUintList(cfg.cnn_conv_kernels, "cnn_conv_kernels");
                }
                if (!cfg.cnn_conv_strides.empty()) {
                    cnnCfg.conv_strides = parseUintList(cfg.cnn_conv_strides, "cnn_conv_strides");
                }
                if (!cfg.cnn_conv_paddings.empty()) {
                    cnnCfg.conv_paddings = parseUintList(cfg.cnn_conv_paddings, "cnn_conv_paddings");
                }
                if (!cfg.cnn_pool_after.empty()) {
                    cnnCfg.pool_after = parseBoolList(cfg.cnn_pool_after, "cnn_pool_after");
                }
                if (!cfg.cnn_pool_kernels.empty()) {
                    cnnCfg.pool_kernels = parseUintList(cfg.cnn_pool_kernels, "cnn_pool_kernels");
                }
                if (!cfg.cnn_pool_strides.empty()) {
                    cnnCfg.pool_strides = parseUintList(cfg.cnn_pool_strides, "cnn_pool_strides");
                }
                if (!cfg.cnn_fc_hidden_sizes.empty()) {
                    cnnCfg.fc_hidden_sizes = parseUintList(cfg.cnn_fc_hidden_sizes, "cnn_fc_hidden_sizes");
                }
                cnnCfg.input_channels = datasetInfo.input_channels;
                cnnCfg.input_height = datasetInfo.input_height;
                cnnCfg.input_width = datasetInfo.input_width;
                cnnCfg.num_classes = datasetInfo.num_classes;
            } catch (const std::exception& e) {
                std::cerr << "Error parsing CNN arguments: " << e.what() << std::endl;
                return 1;
            }
            if (isMaster) {
                std::cout << "Building custom CNN (" << cnnCfg.conv_channels.size()
                          << " conv stages)..." << std::endl;
            }
        } else {
            cnnCfg = CNNConfig::lenet5();
            cnnCfg.input_channels = datasetInfo.input_channels;
            cnnCfg.input_height = datasetInfo.input_height;
            cnnCfg.input_width = datasetInfo.input_width;
            cnnCfg.num_classes = datasetInfo.num_classes;
            if (isMaster) {
                std::cout << "Building LeNet-5 CNN (default)..." << std::endl;
            }
        }
        try {
            model = std::make_unique<CNNNetwork>(cnnCfg, cfg.seed);
        } catch (const std::exception& e) {
            std::cerr << "Error building CNN: " << e.what() << std::endl;
            return 1;
        }
    } else if (cfg.model == "mlp") {
        if (isMaster) {
            std::cout << "Building MLP Network..." << std::endl;
        }
        try {
            auto mlp = std::make_unique<MLPNetwork>();
            if (hiddenVec.size() == 1) {
                *mlp = MLPNetwork::createSingleHidden(
                    static_cast<int>(datasetInfo.input_dim),
                    hiddenVec[0],
                    static_cast<int>(datasetInfo.num_classes),
                    cfg.activation,
                    cfg.init,
                    cfg.seed);
            } else {
                *mlp = MLPNetwork::createMultiHidden(
                    static_cast<int>(datasetInfo.input_dim),
                    hiddenVec,
                    static_cast<int>(datasetInfo.num_classes),
                    cfg.activation,
                    cfg.init,
                    cfg.seed);
            }
            model = std::move(mlp);
        } catch (const std::exception& e) {
            std::cerr << "Error building MLP: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Error: unsupported --model '" << cfg.model
                  << "'. Expected 'mlp' or 'cnn'." << std::endl;
        return 1;
    }

    try {
        // 2. Prepare Data
        if (isMaster) {
            std::cout << "Loading dataset..." << std::endl;
        }
        auto dataset = createDataset(cfg.dataset, cfg.data_dir);
        std::unique_ptr<DataLoader> trainLoader;
        std::unique_ptr<DataLoader> testLoader;

        if (cfg.dataset == "tiny-imagenet") {
            auto* tinyDataset = dynamic_cast<TinyImageNetDataset*>(dataset.get());
            if (tinyDataset == nullptr) {
                throw std::runtime_error("Internal error: tiny-imagenet dataset cast failed.");
            }

            tinyDataset->prepareStreaming();

            if (isMaster) {
                std::cout << "Prepared " << tinyDataset->trainSampleCount()
                          << " training samples and "
                          << tinyDataset->testSampleCount()
                          << " test samples." << std::endl;
            }

            const auto trainIndices = makePaddedRoundRobinIndices(
                tinyDataset->trainSampleCount(), dist.rank(), dist.worldSize());
            const auto testIndices = makeRoundRobinIndices(
                tinyDataset->testSampleCount(), dist.rank(), dist.worldSize());

            if (isMaster && dist.worldSize() > 1) {
                std::cout << "MPI data parallel enabled: local train shard rows = "
                          << trainIndices.size()
                          << ", local test shard rows = " << testIndices.size()
                          << std::endl;
            }

            trainLoader = std::make_unique<DataLoader>(
                tinyDataset->makeTrainBatchSource(trainIndices),
                cfg.batch_size,
                cfg.seed);
            testLoader = std::make_unique<DataLoader>(
                tinyDataset->makeTestBatchSource(testIndices),
                cfg.batch_size,
                cfg.seed);
        } else {
            dataset->load();

            if (isMaster) {
                std::cout << "Loaded " << dataset->getTrainImages().rows << " training samples and "
                          << dataset->getTestImages().rows << " test samples." << std::endl;
            }

            DatasetShard trainShard = makeShard(
                dataset->getTrainImages(),
                dataset->getTrainLabels(),
                makePaddedRoundRobinIndices(
                    dataset->getTrainImages().rows, dist.rank(), dist.worldSize()));
            DatasetShard testShard = makeShard(
                dataset->getTestImages(),
                dataset->getTestLabels(),
                makeRoundRobinIndices(
                    dataset->getTestImages().rows, dist.rank(), dist.worldSize()));

            if (isMaster && dist.worldSize() > 1) {
                std::cout << "MPI data parallel enabled: local train shard rows = "
                          << trainShard.inputs.rows
                          << ", local test shard rows = " << testShard.inputs.rows
                          << std::endl;
            }

            trainLoader = std::make_unique<DataLoader>(
                trainShard.inputs, trainShard.targets, cfg.batch_size, cfg.seed);
            testLoader = std::make_unique<DataLoader>(
                testShard.inputs, testShard.targets, cfg.batch_size, cfg.seed);
        }

        // 3. Setup Loss & Optimizer
        CrossEntropyLoss lossFn;
        std::unique_ptr<Optimizer> optimizer;
        auto params = model->getParameters();
        if (optimizerIsSGD) {
            optimizer = std::make_unique<SGDOptimizer>(std::move(params), cfg.learning_rate);
        } else if (optimizerIsMomentum) {
            optimizer = std::make_unique<MomentumSGDOptimizer>(
                std::move(params),
                cfg.learning_rate,
                cfg.momentum,
                cfg.nesterov,
                cfg.weight_decay
            );
        } else {
            optimizer = std::make_unique<AdamWOptimizer>(
                std::move(params),
                cfg.learning_rate,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay
            );
        }

        Trainer::GradSyncFn gradSyncFn = nullptr;
        if (dist.worldSize() > 1) {
            gradSyncFn = [&dist](const std::vector<Node::Ptr>& paramsToSync,
                                 std::uint64_t localBatch) -> std::uint64_t {
                for (const auto& param : paramsToSync) {
                    Matrix& grad = param->grad();
                    if (!grad.data.empty()) {
                        dist.allReduceSum(grad.data.data(), grad.data.size());
                    }
                }
                return dist.allReduceSumU64(localBatch);
            };
        }

        // 4. Train
        Trainer::ProgressFn progressFn = nullptr;
        if (isMaster) {
            progressFn = [](bool training,
                            std::uint64_t processedBatches,
                            std::uint64_t totalBatches,
                            std::uint64_t processedSamples,
                            std::uint64_t totalSamples) {
                constexpr std::uint64_t kProgressInterval = 50;
                const bool shouldPrint =
                    processedBatches == totalBatches ||
                    processedBatches == 1 ||
                    (processedBatches % kProgressInterval) == 0;
                if (!shouldPrint) {
                    return;
                }

                std::cout << (training ? "[Train progress] " : "[Eval progress] ")
                          << "batch " << processedBatches << "/" << totalBatches
                          << ", samples " << processedSamples << "/" << totalSamples
                          << std::endl;
            };
        }

        Trainer trainer(*model, lossFn, *optimizer, *trainLoader, gradSyncFn, progressFn);

        if (isMaster) {
            std::cout << "Training started..." << std::endl;
        }

        // Create output directory
        if (isMaster) {
            fs::create_directories(cfg.out_dir);
        }

        // Open CSV file for logging
        std::ofstream metricsFile;
        if (isMaster) {
            metricsFile.open(cfg.out_dir + "/metrics.csv");
        }

        if (metricsFile.is_open()) {
            metricsFile
                << "epoch,train_loss,train_acc,test_loss,test_acc,train_samples,"
                << "epoch_time_s,data_time_s,fwd_bwd_time_s,sync_total_time_s,sync_wait_time_s,opt_time_s,"
                << "avg_step_time_ms,max_step_time_ms,samples_per_s,allreduce_wait_ratio,"
                << "world_size,batch_size\n";
        }

        Metrics lastTestMetrics;
        bool hasLastTestMetrics = false;

        for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
            #ifdef PROFILE_MATMUL
            matmulProfileEpochReset();
            #endif

            Metrics trainMetrics = reduceMetrics(dist, trainer.trainEpoch());
            trainMetrics.profile = reduceEpochProfile(dist, trainMetrics.profile);

            // Evaluate on test set with a separate trainer bound to testLoader
            Trainer testTrainer(*model, lossFn, *optimizer, *testLoader, nullptr, progressFn);
            Metrics testMetrics = reduceMetrics(dist, testTrainer.evaluate());
            lastTestMetrics = testMetrics;
            hasLastTestMetrics = true;

            const double avg_step_time_ms =
                (trainMetrics.profile.step_count > 0)
                    ? (trainMetrics.profile.step_time_s_sum /
                       static_cast<double>(trainMetrics.profile.step_count)) * 1000.0
                    : 0.0;
            const double max_step_time_ms = trainMetrics.profile.max_step_time_s * 1000.0;
            const double samples_per_s =
                (trainMetrics.profile.epoch_time_s > 0.0)
                    ? static_cast<double>(trainMetrics.sample_count) / trainMetrics.profile.epoch_time_s
                    : 0.0;
            const double allreduce_wait_ratio =
                (trainMetrics.profile.epoch_time_s > 0.0)
                    ? trainMetrics.profile.sync_wait_time_s / trainMetrics.profile.epoch_time_s
                    : 0.0;

            if (isMaster) {
                std::cout << "Epoch " << epoch << "/" << cfg.epochs
                          << ": [Train] loss = " << std::fixed << std::setprecision(4) << trainMetrics.avg_loss
                          << ", acc = " << std::fixed << std::setprecision(2) << (trainMetrics.accuracy * 100.0) << "%"
                          << " | [Test] loss = " << std::fixed << std::setprecision(4) << testMetrics.avg_loss
                          << ", acc = " << std::fixed << std::setprecision(2) << (testMetrics.accuracy * 100.0) << "%"
                          << " | epoch = " << std::fixed << std::setprecision(3) << trainMetrics.profile.epoch_time_s << "s"
                          << ", data = " << trainMetrics.profile.data_time_s << "s"
                          << ", fwd_bwd = " << trainMetrics.profile.fwd_bwd_time_s << "s"
                          << ", sync_total = " << trainMetrics.profile.sync_total_time_s << "s"
                          << ", sync_wait = " << trainMetrics.profile.sync_wait_time_s << "s"
                          << ", opt = " << trainMetrics.profile.opt_time_s << "s"
                          << ", avg_step = " << avg_step_time_ms << "ms"
                          << ", samples/s = " << samples_per_s
                          << std::endl;
            }

            #ifdef PROFILE_MATMUL
            MatmulEpochStats p = matmulProfileEpochSnapshot();
            if (isMaster) {
                double avg_us = (p.total_calls > 0)
                    ? static_cast<double>(p.total_us) / static_cast<double>(p.total_calls)
                    : 0.0;

                std::cout << "[PROFILE][Epoch " << epoch << "] "
                          << "matmul_calls=" << p.total_calls
                          << ", total_us=" << p.total_us
                          << ", avg_us=" << std::fixed << std::setprecision(2) << avg_us
                          << std::endl;

                for (const auto& s : p.per_impl) {
                    if (s.calls == 0) continue;
                    double impl_avg = static_cast<double>(s.total_us) / static_cast<double>(s.calls);
                    std::cout << "  - " << s.name
                              << ": calls=" << s.calls
                              << ", total_us=" << s.total_us
                              << ", avg_us=" << std::fixed << std::setprecision(2) << impl_avg
                              << std::endl;
                }
            }
            #endif

            if (metricsFile.is_open()) {
                metricsFile << epoch << ","
                            << trainMetrics.avg_loss << "," << trainMetrics.accuracy << ","
                            << testMetrics.avg_loss << "," << testMetrics.accuracy << ","
                            << trainMetrics.sample_count << ","
                            << trainMetrics.profile.epoch_time_s << ","
                            << trainMetrics.profile.data_time_s << ","
                            << trainMetrics.profile.fwd_bwd_time_s << ","
                            << trainMetrics.profile.sync_total_time_s << ","
                            << trainMetrics.profile.sync_wait_time_s << ","
                            << trainMetrics.profile.opt_time_s << ","
                            << avg_step_time_ms << ","
                            << max_step_time_ms << ","
                            << samples_per_s << ","
                            << allreduce_wait_ratio << ","
                            << dist.worldSize() << ","
                            << cfg.batch_size << "\n";
            }
        }

        if (metricsFile.is_open()) {
            metricsFile.close();
        }
        if (isMaster) {
            std::cout << "Training finished. Metrics saved to " << (cfg.out_dir + "/metrics.csv") << std::endl;
        }

        // 5. Final Evaluation
        if (isMaster) {
            std::cout << "\nEvaluating on Test Set..." << std::endl;
        }
        Metrics testMetrics;
        if (cfg.epochs > 0 && hasLastTestMetrics) {
            testMetrics = lastTestMetrics;
        } else {
            Trainer evaluator(*model, lossFn, *optimizer, *testLoader);
            testMetrics = reduceMetrics(dist, evaluator.evaluate());
        }

        if (isMaster) {
            std::cout << "Final Test Accuracy: "
                      << std::fixed << std::setprecision(2) << (testMetrics.accuracy * 100.0) << "%"
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
