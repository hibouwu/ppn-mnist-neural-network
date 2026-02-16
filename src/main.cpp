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

namespace fs = std::filesystem;

#include "mnist_dataset.hpp"
#include "dataloader.hpp"
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
// --------------------------------------------

// configuration struct
struct Config {
    int epochs = 1;                       // Number of training epochs. 0 means "build/eval only, no train loop".
    int batch_size = 64;                  // Mini-batch size used by both train/test DataLoader.
    double learning_rate = 0.01;          // SGD learning rate (> 0).

    // Backward compatible:
    int hidden_size = 128;                // Legacy single hidden layer width for MLP.
    std::string hidden_sizes = "";        // Preferred MLP hidden sizes, e.g. "256,128,64". Empty -> use hidden_size.

    std::string data_dir = "mnist";       // MNIST dataset folder path.
    unsigned int seed = 0;                // RNG seed. 0 means "non-deterministic/random seed path".
    std::string activation = "relu";      // MLP activation: relu / sigmoid / tanh.
    std::string init = "he";              // MLP init strategy: he / xavier / manual.
    std::string model = "mlp";            // Model type: mlp / cnn.

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
            } else if (arg == "--seed") {
                cfg.seed = parseUnsignedIntStrict(requireValue(arg), arg);
            } else if (arg == "--activation") {
                cfg.activation = requireValue(arg);
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

    std::cout << "Starting training with config:" << "\n"
              << "  Epochs: " << cfg.epochs << "\n"
              << "  Batch Size: " << cfg.batch_size << "\n"
              << "  Learning Rate: " << cfg.learning_rate << "\n"
              << "  Hidden (sizes): " << joinHiddenSizes(hiddenVec) << "\n"
              << "  Seed: " << cfg.seed << "\n"
              << "  Activation: " << cfg.activation << "\n"
              << "  Init: " << cfg.init << "\n"
              << "  Model: " << cfg.model << "\n"
              << std::endl;

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
            } catch (const std::exception& e) {
                std::cerr << "Error parsing CNN arguments: " << e.what() << std::endl;
                return 1;
            }
            std::cout << "Building custom CNN (" << cnnCfg.conv_channels.size() << " conv stages)..." << std::endl;
        } else {
            cnnCfg = CNNConfig::lenet5();
            std::cout << "Building LeNet-5 CNN (default)..." << std::endl;
        }
        try {
            model = std::make_unique<CNNNetwork>(cnnCfg, cfg.seed);
        } catch (const std::exception& e) {
            std::cerr << "Error building CNN: " << e.what() << std::endl;
            return 1;
        }
    } else if (cfg.model == "mlp") {
        std::cout << "Building MLP Network..." << std::endl;
        try {
            auto mlp = std::make_unique<MLPNetwork>();
            if (hiddenVec.size() == 1) {
                *mlp = MLPNetwork::createSingleHidden(784, hiddenVec[0], 10, cfg.activation, cfg.init, cfg.seed);
            } else {
                *mlp = MLPNetwork::createMultiHidden(784, hiddenVec, 10, cfg.activation, cfg.init, cfg.seed);
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
        std::cout << "Loading MNIST dataset..." << std::endl;
        MNISTDataset dataset(cfg.data_dir);
        dataset.load();

        std::cout << "Loaded " << dataset.getTrainImages().rows << " training samples and "
                  << dataset.getTestImages().rows << " test samples." << std::endl;

        DataLoader trainLoader(dataset.getTrainImages(), dataset.getTrainLabels(), cfg.batch_size, cfg.seed);
        DataLoader testLoader(dataset.getTestImages(), dataset.getTestLabels(), cfg.batch_size, cfg.seed);

        // 3. Setup Loss & Optimizer
        CrossEntropyLoss lossFn;
        SGDOptimizer optimizer(model->getParameters(), cfg.learning_rate);

        // 4. Train
        Trainer trainer(*model, lossFn, optimizer, trainLoader);

        std::cout << "Training started..." << std::endl;

        // Create output directory
        fs::create_directories(cfg.out_dir);

        // Open CSV file for logging
        std::ofstream metricsFile(cfg.out_dir + "/metrics.csv");

        if (metricsFile.is_open()) {
            metricsFile << "epoch,train_loss,train_acc,test_loss,test_acc\n";
        }

        for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
            #ifdef PROFILE_MATMUL
            matmulProfileEpochReset();
            #endif

            Metrics trainMetrics = trainer.trainEpoch();

            // Evaluate on test set with a separate trainer bound to testLoader
            Trainer testTrainer(*model, lossFn, optimizer, testLoader);
            Metrics testMetrics = testTrainer.evaluate();

            std::cout << "Epoch " << epoch << "/" << cfg.epochs
                      << ": [Train] loss = " << std::fixed << std::setprecision(4) << trainMetrics.avg_loss
                      << ", acc = " << std::fixed << std::setprecision(2) << (trainMetrics.accuracy * 100.0) << "%"
                      << " | [Test] loss = " << std::fixed << std::setprecision(4) << testMetrics.avg_loss
                      << ", acc = " << std::fixed << std::setprecision(2) << (testMetrics.accuracy * 100.0) << "%"
                      << std::endl;

            #ifdef PROFILE_MATMUL
            MatmulEpochStats p = matmulProfileEpochSnapshot();
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
            #endif

            if (metricsFile.is_open()) {
                metricsFile << epoch << ","
                            << trainMetrics.avg_loss << "," << trainMetrics.accuracy << ","
                            << testMetrics.avg_loss << "," << testMetrics.accuracy << "\n";
            }
        }

        metricsFile.close();
        std::cout << "Training finished. Metrics saved to " << (cfg.out_dir + "/metrics.csv") << std::endl;

        // 5. Final Evaluation
        std::cout << "\nEvaluating on Test Set..." << std::endl;
        Trainer evaluator(*model, lossFn, optimizer, testLoader);
        Metrics testMetrics = evaluator.evaluate();

        std::cout << "Final Test Accuracy: "
                  << std::fixed << std::setprecision(2) << (testMetrics.accuracy * 100.0) << "%"
                  << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
