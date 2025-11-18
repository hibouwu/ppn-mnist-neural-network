#include "node.hpp"
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

// Constructeur : stocke la valeur et crée un gradient de même taille initialisé à zéro
Node::Node(const Matrix& v)
    : value_(v), grad_(v.rows, v.cols, 0.0) {}

// Ajoute un gradient au gradient actuel : grad_ += g
void Node::addGrad(const Matrix& g) {
    if (g.rows != grad_.rows || g.cols != grad_.cols) {
        throw std::invalid_argument("Dimensions incompatibles dans addGrad.");
    }
    const std::size_t n = grad_.data.size();
    for (std::size_t i = 0; i < n; ++i) {
        grad_.data[i] += g.data[i];
    }
}

// Met le gradient à zéro
void Node::zeroGrad() {
    std::fill(grad_.data.begin(), grad_.data.end(), 0.0);
}

// Effectue un tri topologique en DFS : parents d'abord, puis le nœud courant
std::vector<Node::Ptr> Node::topoSort(const Ptr& root) {
    std::vector<Ptr> order;
    std::unordered_set<const Node*> vis;

    std::function<void(const Ptr&)> dfs = [&](const Ptr& u) {
        if (!u || vis.count(u.get())) return;
        vis.insert(u.get());

        // Visite des parents en premier
        for (auto &wp : u->parents_) {
            if (auto p = wp.lock()) dfs(p);
        }

        // Ajout du nœud après ses parents
        order.push_back(u);
    };

    dfs(root);
    return order;  // ordre : [parents ..., root]
}

// Lance la rétropropagation du gradient
void Node::backward() {
    auto self  = shared_from_this();
    auto order = topoSort(self);  // ordre : parents → enfants

    // Si le gradient du nœud racine est entièrement nul, on le remplace par 1
    Matrix &rootGrad = self->grad();
    bool allZero = true;
    for (double v : rootGrad.data) {
        if (v != 0.0) { allZero = false; break; }
    }
    if (allZero) {
        for (double &v : rootGrad.data) v = 1.0;  // gradient initial
    }

    // Rétropropagation en ordre inverse : root → parents
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        auto &n = *it;
        if (n->backwardFn_) {
            n->backwardFn_(n->grad());
        }
    }
}

