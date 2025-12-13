#pragma once

#include <memory>
#include "node.hpp"

// Interface de fonction de perte
class LossFunction {
public:
    using NodePtr = std::shared_ptr<Node>;
    virtual ~LossFunction() = default;

    // pred : sortie du modèle (souvent logits ou prédictions)
    // target : labels (souvent one-hot)
    // Retour : un nœud scalaire (1x1) représentant la perte du batch
    virtual NodePtr forward(const NodePtr& pred, const NodePtr& target) = 0;
};

// MSE : somme sur le batch de la MSE par échantillon
// L = (1 / C) * sum_{i=1..B} sum_{j=1..C} (pred_ij - target_ij)^2
class MSELoss : public LossFunction {
public:
    NodePtr forward(const NodePtr& pred, const NodePtr& target) override;
};

// Cross Entropy sur logits + softmax interne (stable)
// target est supposé one-hot (ou distribution)
// L = sum_{i=1..B} - sum_{c=1..C} y_ic * log softmax(logits)_ic
class CrossEntropyLoss : public LossFunction {
public:
    explicit CrossEntropyLoss(double eps = 1e-12) : eps_(eps) {}
    NodePtr forward(const NodePtr& pred, const NodePtr& target) override;

private:
    double eps_;
};
