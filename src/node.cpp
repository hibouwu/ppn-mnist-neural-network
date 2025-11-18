#include "node.hpp"
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

Node::Node(const Matrix& v)
    : value_(v), grad_(v.rows, v.cols, 0.0) {}

void Node::addGrad(const Matrix& g) {
    if (g.rows != grad_.rows || g.cols != grad_.cols) {
        throw std::invalid_argument("Gradient shape mismatch in Node::addGrad");
    }
    const std::size_t n = grad_.data.size();
    for (std::size_t i = 0; i < n; ++i) {
        grad_.data[i] += g.data[i];
    }
}

void Node::zeroGrad() {
    std::fill(grad_.data.begin(), grad_.data.end(), 0.0);
}

std::vector<Node::Ptr> Node::topoSort(const Ptr& root) {
    std::vector<Ptr> order;
    std::unordered_set<const Node*> vis;

    std::function<void(const Ptr&)> dfs = [&](const Ptr& u) {
        if (!u || vis.count(u.get())) return;
        vis.insert(u.get());

        // Traversez d'abord le nœud parent, puis insérez-vous dans la file d'attente.
        for (auto &wp : u->parents_) {
            if (auto p = wp.lock()) dfs(p);
        }
        order.push_back(u);
    };

    dfs(root);
    // Comme nous effectuons d'abord la récursivité du parent, nous nous ajoutons ensuite à la pile.
    // L'ordre obtenu est intrinsèquement le suivant : tous les nœuds parents d'abord, suivis des nœuds enfants, ce qui ne nécessite donc aucun renversement.
    return order;
}

void Node::backward() {
    auto self  = shared_from_this();
    auto order = topoSort(self);

    // 1) Réinitialiser tous les dégradés
    for (auto &n : order) n->zeroGrad();

    // 2) Injection de gradients de graines dans le nœud racine :
    //    Scalaire -> 1 ; sinon -> tous les 1
    if (value_.rows == 1 && value_.cols == 1)
        order.back()->grad_ = Matrix(1, 1, 1.0);
    else
        order.back()->grad_ = Matrix(value_.rows, value_.cols, 1.0);

    // 3) Exécutez la fonction arrière propre à chaque nœud dans l'ordre topologique.
    for (auto &n : order) {
        if (n->backwardFn_) n->backwardFn_(n->grad_);
    }
}


