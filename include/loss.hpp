#pragma once

#include <memory>
#include "node.hpp"

// Module de fonctions de perte
namespace Loss {

using NodePtr = std::shared_ptr<Node>;

// MSELoss(pred, target) = mean( (pred - target)^2 )
NodePtr MSELoss(const NodePtr& pred, const NodePtr& target);

// CrossEntropyLoss(pred, target)
// - pred : probabilités (par ex. sortie de softmax)
// - target : étiquettes one-hot ou distributions
NodePtr CrossEntropyLoss(const NodePtr& pred,
                         const NodePtr& target,
                         double eps = 1e-12);

} // namespace Loss
