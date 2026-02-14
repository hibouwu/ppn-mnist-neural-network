#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

#include "mnist_dataset.hpp"
#include "dataloader.hpp"
#include "network.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "trainer.hpp"
#include "activation.hpp"

#include "profiling.hpp"

// ------------------ helpers ------------------
static std::vector<int> parseHiddenSizes(const std::string& s) {
    std::vector<int> hs;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) hs.push_back(std::stoi(item));
    }
    return hs;
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
    int epochs = 1;
    int batch_size = 64;
    double learning_rate = 0.01;

    // Backward compatible:
    int hidden_size = 128;              // old single-hidden param
    std::string hidden_sizes = "";      // new multi-hidden param: "256,128,64"

    std::string data_dir = "mnist";
    unsigned int seed = 0; // 0 = random
    std::string activation = "relu"; // relu, sigmoid, tanh
    std::string init = "he"; // he, xavier, manual

    std::string out_dir = "output";//optuna

};

int main(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--epochs" && i + 1 < argc) {
            cfg.epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch_size" && i + 1 < argc) {
            cfg.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--learning_rate" && i + 1 < argc) {
            cfg.learning_rate = std::stod(argv[++i]);
        } else if (arg == "--hidden_size" && i + 1 < argc) {
            cfg.hidden_size = std::stoi(argv[++i]);
        } else if (arg == "--hidden_sizes" && i + 1 < argc) {
            cfg.hidden_sizes = argv[++i];
        } else if (arg == "--data_dir" && i + 1 < argc) {
            cfg.data_dir = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            cfg.seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        } else if (arg == "--activation" && i + 1 < argc) {
            cfg.activation = argv[++i];
        } else if (arg == "--init" && i + 1 < argc) {
            cfg.init = argv[++i];
        } else if (arg == "--out_dir" && i + 1 < argc) {
            cfg.out_dir = argv[++i];  // optuna
}

    }

    // Decide architecture
    std::vector<int> hiddenVec;
    if (!cfg.hidden_sizes.empty()) {
        hiddenVec = parseHiddenSizes(cfg.hidden_sizes);
    } else {
        hiddenVec = { cfg.hidden_size };
    }

    std::cout << "Starting training with config:" << "\n"
              << "  Epochs: " << cfg.epochs << "\n"
              << "  Batch Size: " << cfg.batch_size << "\n"
              << "  Learning Rate: " << cfg.learning_rate << "\n"
              << "  Hidden (sizes): " << joinHiddenSizes(hiddenVec) << "\n"
              << "  Seed: " << cfg.seed << "\n"
              << "  Activation: " << cfg.activation << "\n"
              << "  Init: " << cfg.init << "\n"
              << std::endl;

    // 1. Prepare Data
    std::cout << "Loading MNIST dataset..." << std::endl;
    MNISTDataset dataset(cfg.data_dir);
    dataset.load();

    std::cout << "Loaded " << dataset.getTrainImages().rows << " training samples and "
              << dataset.getTestImages().rows << " test samples." << std::endl;

    DataLoader trainLoader(dataset.getTrainImages(), dataset.getTrainLabels(), cfg.batch_size, cfg.seed);
    DataLoader testLoader(dataset.getTestImages(), dataset.getTestLabels(), cfg.batch_size, cfg.seed);

    // 2. Build Model
    std::cout << "Building MLP Network..." << std::endl;

    MLPNetwork model;
    if (hiddenVec.size() == 1) {
        model = MLPNetwork::createSingleHidden(784, hiddenVec[0], 10, cfg.activation, cfg.init, cfg.seed);
    } else {
        model = MLPNetwork::createMultiHidden(784, hiddenVec, 10, cfg.activation, cfg.init, cfg.seed);
    }

    // 3. Setup Loss & Optimizer
    CrossEntropyLoss lossFn;
    SGDOptimizer optimizer(model.getParameters(), cfg.learning_rate);

    // 4. Train
    Trainer trainer(model, lossFn, optimizer, trainLoader);

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
        Trainer testTrainer(model, lossFn, optimizer, testLoader);
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
    Trainer evaluator(model, lossFn, optimizer, testLoader);
    Metrics testMetrics = evaluator.evaluate();

    std::cout << "Final Test Accuracy: "
              << std::fixed << std::setprecision(2) << (testMetrics.accuracy * 100.0) << "%"
              << std::endl;

    return 0;
}
