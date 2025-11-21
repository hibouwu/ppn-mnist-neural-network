#ifndef LAYER_HPP
#define LAYER_HPP

#include "node.hpp"

class LinearLayer {
public:
    // Constructeur
    LinearLayer(size_t in, size_t out);

    // Propagation avant
    Node::Ptr forward(const Node::Ptr& input) const;

    // Initialisation aléatoire
    void randomInit(double min = -1.0, double max = 1.0);

    // Récupérer les paramètres (pour l'optimiseur)
    std::vector<Node::Ptr> parameters() const;

private:
    size_t in_dim;
    size_t out_dim;
    Node::Ptr weights_;  // Shape: (in_dim, out_dim)
    Node::Ptr bias_;     // Shape: (1, out_dim)
};

#endif
