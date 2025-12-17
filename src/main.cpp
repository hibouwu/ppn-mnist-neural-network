#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem> // Added for directory creation

namespace fs = std::filesystem;
#include "mnist_dataset.hpp"
#include "dataloader.hpp"
#include "network.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "trainer.hpp"
#include "activation.hpp"

// Simple configuration struct
struct Config {
    int epochs = 1;
    int batch_size = 64;
    double learning_rate = 0.01;
    int hidden_size = 128;
    std::string data_dir = "mnist";
};

int main(int argc, char** argv) {
    Config cfg;
    if (argc > 1) {
        try {
            cfg.epochs = std::stoi(argv[1]);
        } catch (...) {
            std::cerr << "Invalid epoch argument, using default: " << cfg.epochs << std::endl;
        }
    }
    std::cout << "Starting training with config:" << "\n"
              << "  Epochs: " << cfg.epochs << "\n"
              << "  Batch Size: " << cfg.batch_size << "\n"
              << "  Learning Rate: " << cfg.learning_rate << "\n"
              << "  Hidden Size: " << cfg.hidden_size << "\n"
              << std::endl;

    // 1. Prepare Data
    std::cout << "Loading MNIST dataset..." << std::endl;
    MNISTDataset dataset(cfg.data_dir);
    dataset.load(); // Loads both train and test
    
    std::cout << "Loaded " << dataset.getTrainImages().rows << " training samples and " 
              << dataset.getTestImages().rows << " test samples." << std::endl;

    DataLoader trainLoader(dataset.getTrainImages(), dataset.getTrainLabels(), cfg.batch_size);
    DataLoader testLoader(dataset.getTestImages(), dataset.getTestLabels(), cfg.batch_size); // Keep testLoader for later use

    // 2. Build Model (784 -> 128 -> 10)
    std::cout << "Building MLP Network..." << std::endl;
    MLPNetwork model;
    
    // Layer 1: 784 -> 128 + ReLU
    auto fc1 = std::make_unique<LinearLayer>(784, cfg.hidden_size);
    fc1->randomInit(-0.1, 0.1);
    model.addLayer(std::move(fc1), std::make_unique<ReLU>());

    // Layer 2: 128 -> 10 + Sigmoid (or Softmax implicitly via CrossEntropy)
    // Note: Our CrossEntropyLoss expects LOGITS (raw outputs), so we usually 
    // don't put an activation on the last layer, OR we put Identity.
    // However, the current MLP structure enforces an activation pair. 
    // Let's check network.hpp. It uses LayerNode {Linear, Activation}.
    // If we want raw logits, we might need an Identity activation.
    // BUT, for now let's use Sigmoid or Tanh if Identity is missing, 
    // OR just verify if CrossEntropy handles probabilities.
    // CrossEntropyLoss implementation: "CrossEntropyLoss(pred, target)"
    // It implemented "stable softmax" internally? 
    // Let's re-read src/loss.cpp carefully.
    // Yes, Step 214 shows CrossEntropyLoss computes softmax internally on `pred`.
    // So `pred` should be partial logits.
    // But MLPNetwork::forward iterates layers and applies activation.
    // If we put Sigmoid at the end, `pred` will be in [0,1].
    // Then CrossEntropy performing softmax on [0,1] values is mathematically generic but maybe not what we want (double activation).
    // Standard practice: Last layer is Linear only (Identity activation).
    // Does our framework support Identity activation?
    // Checking activation.hpp... ReLU, Sigmoid, Tanh. No Identity.
    // To be strictly correct, we should add Identity. 
    // For now, I will use Sigmoid, and acknowledge it's sub-optimal but works (Stacked Softmax/Sigmoid).
    // Actually, let's just add a simple Identity class in main locally or use Sigmoid.
    // Using Sigmoid is safe-ish, just squashes logits.
    
    auto fc2 = std::make_unique<LinearLayer>(cfg.hidden_size, 10);
    fc2->randomInit(-0.1, 0.1);
    // Ideally use Identity, here passing nullptr means no activation (Raw Logits)
    model.addLayer(std::move(fc2), nullptr);

    // 3. Setup Loss & Optimizer
    CrossEntropyLoss lossFn;
    SGDOptimizer optimizer(model.getParameters(), cfg.learning_rate);

    // 4. Train
    // --- 5. Training Loop ---
    Trainer trainer(model, lossFn, optimizer, trainLoader); // Use trainLoader here
    std::cout << "Training started..." << std::endl;

    // Create output directory
    fs::create_directory("output");

    // Open CSV file for logging
    std::ofstream metricsFile("output/metrics.csv");
    if (metricsFile.is_open()) {
        metricsFile << "epoch,train_loss,train_acc,test_loss,test_acc\n";
    }

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        Metrics trainMetrics = trainer.trainEpoch();
        
        // Evaluate on test set
        // To do Test Evaluation properly:
        // We need a DataLoader for the Test Set.
        // The current configuration:
        // MNISTDataset dataset ...
        // DataLoader dataLoader(dataset.getTrainImages(), dataset.getTrainLabels(), config.batch_size);
        // Trainer trainer(..., dataLoader, ...);
        
        // Problem: Trainer is bound to Train DataLoader.
        // Trainer::evaluate() just calls `runEpoch(false)`. 
        // `runEpoch` uses `dataLoader_`.
        // So `trainer.evaluate()` calculates metrics on the TRAIN set (without backprop).
        
        // To fix this without major refactor:
        // We can create a second DataLoader for test:
        // DataLoader testLoader(dataset.getTestImages(), dataset.getTestLabels(), config.batch_size);
        // And we need a way to evaluate using `testLoader`.
        // Since Trainer doesn't support swapping loader, we can manually create a temporary Trainer 
        // OR just instantiate another Trainer for testing?
        // `Trainer testTrainer(model, loss, optimizer, testLoader, config);`
        // `Metrics testMetrics = testTrainer.evaluate();`
        // This is safe because `evaluate` doesn't modify the model (no optimizer step).
        
        // Use the already defined testLoader
        Trainer testTrainer(model, lossFn, optimizer, testLoader); // Pass testLoader
        Metrics testMetrics = testTrainer.evaluate();

        std::cout << "Epoch " << epoch << "/" << cfg.epochs 
                  << ": [Train] loss = " << std::fixed << std::setprecision(4) << trainMetrics.avg_loss 
                  << ", acc = " << std::fixed << std::setprecision(2) << (trainMetrics.accuracy * 100.0) << "%"
                  << " | [Test] loss = " << std::fixed << std::setprecision(4) << testMetrics.avg_loss 
                  << ", acc = " << std::fixed << std::setprecision(2) << (testMetrics.accuracy * 100.0) << "%" << std::endl;

        if (metricsFile.is_open()) {
            metricsFile << epoch << "," 
                        << trainMetrics.avg_loss << "," << trainMetrics.accuracy << ","
                        << testMetrics.avg_loss << "," << testMetrics.accuracy << "\n";
        }
    }
    
    metricsFile.close();
    std::cout << "Training finished. Metrics saved to metrics.csv" << std::endl;


    // 5. Final Evaluation Logic (Manual since Trainer is bound to trainLoader)
    std::cout << "\nEvaluating on Test Set..." << std::endl;
    // The final evaluation is now part of the loop and saved to CSV.
    // The original "Manual" evaluation block is redundant if the loop already does it.
    // However, to match the instruction's structure, I'll keep the final evaluation block,
    // but it will essentially re-evaluate the test set one more time.
    // The instruction's provided snippet ends with `{{ ... }}` implying the rest of the file remains.
    
    // Hack: Create a new Trainer just for evaluation?
    // Or just run manual loop. 
    // Let's create a temporary evaluator trainer.
    // Note: Optimizer is not used in eval, pass dummy or same.
    Trainer evaluator(model, lossFn, optimizer, testLoader);
    Metrics testMetrics = evaluator.evaluate();
    
    std::cout << "Final Test Accuracy: " 
              << std::fixed << std::setprecision(2) << (testMetrics.accuracy * 100.0) << "%" << std::endl;

    return 0;
}
