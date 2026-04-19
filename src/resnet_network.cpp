/**
 * @file resnet_network.cpp
 * @brief Small serial ResNet network implementation.
 */
#include "resnet_network.hpp"

#include "math_ops.hpp"

#include <stdexcept>

namespace {

static LinearLayer::InitType parseInit(const std::string& init_name) {
    if (init_name == "he") return LinearLayer::InitType::He;
    if (init_name == "xavier") return LinearLayer::InitType::Xavier;
    return LinearLayer::InitType::Manual;
}

static std::unique_ptr<ActivationFunction> makeActivation(const std::string& name) {
    if (name == "relu")       return std::make_unique<ReLU>();
    if (name == "leaky_relu") return std::make_unique<LeakyReLU>();
    if (name == "gelu")       return std::make_unique<GELU>();
    if (name == "sigmoid")    return std::make_unique<Sigmoid>();
    if (name == "tanh")       return std::make_unique<Tanh>();
    throw std::invalid_argument(
        "unsupported activation '" + name +
        "'. Expected one of: relu, leaky_relu, gelu, sigmoid, tanh.");
}

static void advanceSeed(unsigned int& seed) {
    if (seed != 0) {
        ++seed;
    }
}

}  // namespace

void ResNetConfig::validate() const {
    if (input_channels == 0 || input_height == 0 || input_width == 0) {
        throw std::invalid_argument("ResNetConfig: input shape must be strictly positive.");
    }
    if (stem_channels == 0) {
        throw std::invalid_argument("ResNetConfig: stem_channels must be > 0.");
    }
    if (stage_channels.empty()) {
        throw std::invalid_argument("ResNetConfig: stage_channels must not be empty.");
    }
    if (stage_channels.size() != blocks_per_stage.size()) {
        throw std::invalid_argument("ResNetConfig: stage_channels and blocks_per_stage size mismatch.");
    }
    for (size_t i = 0; i < stage_channels.size(); ++i) {
        if (stage_channels[i] == 0) {
            throw std::invalid_argument("ResNetConfig: stage_channels entries must be > 0.");
        }
        if (blocks_per_stage[i] == 0) {
            throw std::invalid_argument("ResNetConfig: blocks_per_stage entries must be > 0.");
        }
    }
    if (num_classes == 0) {
        throw std::invalid_argument("ResNetConfig: num_classes must be > 0.");
    }
}

ResNetNetwork::BasicBlock::BasicBlock(size_t in_channels,
                                      size_t out_channels,
                                      size_t stride,
                                      ConvBackend backend)
    : conv1_(in_channels, out_channels, 3, 3, stride, 1, backend),
      conv2_(out_channels, out_channels, 3, 3, 1, 1, backend),
      in_channels_(in_channels),
      out_channels_(out_channels),
      stride_(stride)
{
    if (stride == 0) {
        throw std::invalid_argument("ResNet BasicBlock: stride must be > 0.");
    }
    if (stride != 1 || in_channels != out_channels) {
        projection_ = std::make_unique<Conv2DLayer>(
            in_channels, out_channels, 1, 1, stride, 0, backend);
    }
}

std::pair<size_t, size_t> ResNetNetwork::BasicBlock::outputShape(size_t H, size_t W) const {
    auto [h1, w1] = conv1_.outputShape(H, W);
    return conv2_.outputShape(h1, w1);
}

Node::Ptr ResNetNetwork::BasicBlock::forward(const Node::Ptr& input,
                                             size_t N,
                                             size_t& C,
                                             size_t& H,
                                             size_t& W,
                                             const ActivationFunction& activation) const {
    const size_t inC = C;
    const size_t inH = H;
    const size_t inW = W;

    auto out = conv1_.forward(input, N, inC, inH, inW);
    out = activation.forward(out);
    auto [h1, w1] = conv1_.outputShape(inH, inW);

    out = conv2_.forward(out, N, out_channels_, h1, w1);
    auto [h2, w2] = conv2_.outputShape(h1, w1);

    Node::Ptr shortcut = input;
    if (projection_) {
        shortcut = projection_->forward(input, N, inC, inH, inW);
    }

    out = MathOps::add(out, shortcut);
    out = activation.forward(out);

    C = out_channels_;
    H = h2;
    W = w2;
    return out;
}

void ResNetNetwork::BasicBlock::randomInit(unsigned int& seed) {
    conv1_.randomInit(seed);
    advanceSeed(seed);
    conv2_.randomInit(seed);
    advanceSeed(seed);
    if (projection_) {
        projection_->randomInit(seed);
        advanceSeed(seed);
    }
}

std::vector<Node::Ptr> ResNetNetwork::BasicBlock::parameters() const {
    std::vector<Node::Ptr> params;
    auto conv1_params = conv1_.parameters();
    params.insert(params.end(), conv1_params.begin(), conv1_params.end());
    auto conv2_params = conv2_.parameters();
    params.insert(params.end(), conv2_params.begin(), conv2_params.end());
    if (projection_) {
        auto projection_params = projection_->parameters();
        params.insert(params.end(), projection_params.begin(), projection_params.end());
    }
    return params;
}

ResNetNetwork::ResNetNetwork(const ResNetConfig& cfg,
                             const std::string& activation_name,
                             const std::string& init_name,
                             unsigned int seed)
    : stem_(cfg.input_channels, cfg.stem_channels, 3, 3, 1, 1, cfg.conv_backend),
      classifier_(1, cfg.num_classes, false),
      activation_(makeActivation(activation_name)),
      init_name_(init_name),
      input_channels_(cfg.input_channels),
      input_height_(cfg.input_height),
      input_width_(cfg.input_width),
      input_dim_(cfg.input_channels * cfg.input_height * cfg.input_width),
      flatten_dim_(0)
{
    cfg.validate();

    size_t C = cfg.stem_channels;
    auto [H, W] = stem_.outputShape(input_height_, input_width_);
    if (H == 0 || W == 0) {
        throw std::invalid_argument("ResNetNetwork: stem output shape is 0.");
    }

    size_t prev_channels = cfg.stem_channels;
    for (size_t stage = 0; stage < cfg.stage_channels.size(); ++stage) {
        const size_t out_channels = cfg.stage_channels[stage];
        for (size_t block = 0; block < cfg.blocks_per_stage[stage]; ++block) {
            const bool downsample = (stage > 0 && block == 0);
            const size_t stride = downsample ? 2 : 1;
            blocks_.emplace_back(prev_channels, out_channels, stride, cfg.conv_backend);

            auto [nextH, nextW] = blocks_.back().outputShape(H, W);
            if (nextH == 0 || nextW == 0) {
                throw std::invalid_argument(
                    "ResNetNetwork: residual block output shape is 0.");
            }
            H = nextH;
            W = nextW;
            C = out_channels;
            prev_channels = out_channels;
        }
    }

    flatten_dim_ = C * H * W;
    classifier_ = LinearLayer(flatten_dim_, cfg.num_classes, false);

    unsigned int s = seed;
    stem_.randomInit(s);
    advanceSeed(s);
    for (auto& block : blocks_) {
        block.randomInit(s);
    }

    classifier_.randomInit(0.0, 0.0, parseInit(init_name_), s);
}

ResNetNetwork::ResNetNetwork(const ResNetConfig& cfg, unsigned int seed)
    : ResNetNetwork(cfg, "relu", "he", seed)
{}

Node::Ptr ResNetNetwork::forward(const Node::Ptr& input) const {
    if (!input) {
        throw std::invalid_argument("ResNetNetwork::forward: input node is null.");
    }
    const Matrix& in = input->value();
    if (in.cols != input_dim_) {
        throw std::invalid_argument(
            "ResNetNetwork::forward: expected input (N, " + std::to_string(input_dim_) + ").");
    }

    size_t N = in.rows;
    size_t C = input_channels_;
    size_t H = input_height_;
    size_t W = input_width_;

    auto x = stem_.forward(input, N, C, H, W);
    x = activation_->forward(x);
    auto [stemH, stemW] = stem_.outputShape(H, W);
    C = stem_.outChannels();
    H = stemH;
    W = stemW;

    for (const auto& block : blocks_) {
        x = block.forward(x, N, C, H, W, *activation_);
    }

    return classifier_.forward(x);
}

std::vector<Node::Ptr> ResNetNetwork::getParameters() const {
    std::vector<Node::Ptr> params;
    auto stem_params = stem_.parameters();
    params.insert(params.end(), stem_params.begin(), stem_params.end());
    for (const auto& block : blocks_) {
        auto block_params = block.parameters();
        params.insert(params.end(), block_params.begin(), block_params.end());
    }
    auto classifier_params = classifier_.parameters();
    params.insert(params.end(), classifier_params.begin(), classifier_params.end());
    return params;
}
