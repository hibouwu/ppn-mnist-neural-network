#include "checkpoint.hpp"

#include "optimizer.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace checkpoint {
namespace {

constexpr char kCheckpointMagic[] = "PPNCHKPT";
constexpr std::uint32_t kCheckpointVersion = 1;

template <typename T>
void writeExact(std::ostream& os, const T& value, const char* what) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!os) {
        throw std::runtime_error(std::string("Failed to write ") + what + ".");
    }
}

template <typename T>
void readExact(std::istream& is, T& value, const char* what) {
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!is) {
        throw std::runtime_error(std::string("Failed to read ") + what + ".");
    }
}

void writeString(std::ostream& os, const std::string& value, const char* what) {
    const std::uint64_t size = static_cast<std::uint64_t>(value.size());
    writeExact(os, size, what);
    if (size > 0) {
        os.write(value.data(), static_cast<std::streamsize>(size));
        if (!os) {
            throw std::runtime_error(std::string("Failed to write ") + what + ".");
        }
    }
}

std::string readString(std::istream& is, const char* what) {
    std::uint64_t size = 0;
    readExact(is, size, what);
    std::string value(static_cast<std::size_t>(size), '\0');
    if (size > 0) {
        is.read(&value[0], static_cast<std::streamsize>(size));
        if (!is) {
            throw std::runtime_error(std::string("Failed to read ") + what + ".");
        }
    }
    return value;
}

void writeBool(std::ostream& os, bool value, const char* what) {
    const std::uint8_t byte = value ? 1u : 0u;
    writeExact(os, byte, what);
}

bool readBool(std::istream& is, const char* what) {
    std::uint8_t byte = 0;
    readExact(is, byte, what);
    if (byte > 1u) {
        throw std::runtime_error(std::string("Invalid boolean value for ") + what + ".");
    }
    return byte != 0u;
}

void writeMatrix(std::ostream& os, const Matrix& matrix, const char* what) {
    const std::uint64_t rows = static_cast<std::uint64_t>(matrix.rows);
    const std::uint64_t cols = static_cast<std::uint64_t>(matrix.cols);
    const std::uint64_t size = static_cast<std::uint64_t>(matrix.data.size());
    writeExact(os, rows, what);
    writeExact(os, cols, what);
    writeExact(os, size, what);
    if (size > 0) {
        os.write(reinterpret_cast<const char*>(matrix.data.data()),
                 static_cast<std::streamsize>(size * sizeof(Scalar)));
        if (!os) {
            throw std::runtime_error(std::string("Failed to write ") + what + ".");
        }
    }
}

Matrix readMatrix(std::istream& is, const char* what) {
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t size = 0;
    readExact(is, rows, what);
    readExact(is, cols, what);
    readExact(is, size, what);
    if (rows == 0 || cols == 0) {
        throw std::runtime_error(std::string("Invalid matrix shape for ") + what + ".");
    }
    if (size != rows * cols) {
        throw std::runtime_error(std::string("Invalid matrix size for ") + what + ".");
    }

    Matrix matrix(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
    if (size > 0) {
        is.read(reinterpret_cast<char*>(matrix.data.data()),
                static_cast<std::streamsize>(size * sizeof(Scalar)));
        if (!is) {
            throw std::runtime_error(std::string("Failed to read ") + what + ".");
        }
    }
    return matrix;
}

void writeMetadata(std::ostream& os, const Metadata& metadata) {
    writeExact(os, metadata.epoch, "checkpoint epoch");
    writeExact(os, static_cast<std::int32_t>(metadata.batch_size), "checkpoint batch_size");
    writeExact(os, metadata.learning_rate, "checkpoint learning_rate");
    writeExact(os, static_cast<std::uint32_t>(metadata.seed), "checkpoint seed");

    writeString(os, metadata.dataset, "checkpoint dataset");
    writeString(os, metadata.data_dir, "checkpoint data_dir");
    writeString(os, metadata.model, "checkpoint model");
    writeString(os, metadata.activation, "checkpoint activation");
    writeString(os, metadata.init, "checkpoint init");
    writeString(os, metadata.optimizer, "checkpoint optimizer");
    writeString(os, metadata.hidden_sizes, "checkpoint hidden_sizes");

    writeExact(os, metadata.momentum, "checkpoint momentum");
    writeBool(os, metadata.nesterov, "checkpoint nesterov");
    writeExact(os, metadata.weight_decay, "checkpoint weight_decay");
    writeExact(os, metadata.beta1, "checkpoint beta1");
    writeExact(os, metadata.beta2, "checkpoint beta2");
    writeExact(os, metadata.eps, "checkpoint eps");

    writeString(os, metadata.cnn_conv_channels, "checkpoint cnn_conv_channels");
    writeString(os, metadata.cnn_conv_kernels, "checkpoint cnn_conv_kernels");
    writeString(os, metadata.cnn_conv_strides, "checkpoint cnn_conv_strides");
    writeString(os, metadata.cnn_conv_paddings, "checkpoint cnn_conv_paddings");
    writeString(os, metadata.cnn_pool_after, "checkpoint cnn_pool_after");
    writeString(os, metadata.cnn_pool_kernels, "checkpoint cnn_pool_kernels");
    writeString(os, metadata.cnn_pool_strides, "checkpoint cnn_pool_strides");
    writeString(os, metadata.cnn_fc_hidden_sizes, "checkpoint cnn_fc_hidden_sizes");
    writeString(os, metadata.conv_backend, "checkpoint conv_backend");
}

Metadata readMetadata(std::istream& is) {
    Metadata metadata;
    metadata.epoch = 0;

    std::int32_t batch_size = 0;
    std::uint32_t seed = 0;
    readExact(is, metadata.epoch, "checkpoint epoch");
    readExact(is, batch_size, "checkpoint batch_size");
    readExact(is, metadata.learning_rate, "checkpoint learning_rate");
    readExact(is, seed, "checkpoint seed");
    metadata.batch_size = static_cast<int>(batch_size);
    metadata.seed = static_cast<unsigned int>(seed);

    metadata.dataset = readString(is, "checkpoint dataset");
    metadata.data_dir = readString(is, "checkpoint data_dir");
    metadata.model = readString(is, "checkpoint model");
    metadata.activation = readString(is, "checkpoint activation");
    metadata.init = readString(is, "checkpoint init");
    metadata.optimizer = readString(is, "checkpoint optimizer");
    metadata.hidden_sizes = readString(is, "checkpoint hidden_sizes");

    readExact(is, metadata.momentum, "checkpoint momentum");
    metadata.nesterov = readBool(is, "checkpoint nesterov");
    readExact(is, metadata.weight_decay, "checkpoint weight_decay");
    readExact(is, metadata.beta1, "checkpoint beta1");
    readExact(is, metadata.beta2, "checkpoint beta2");
    readExact(is, metadata.eps, "checkpoint eps");

    metadata.cnn_conv_channels = readString(is, "checkpoint cnn_conv_channels");
    metadata.cnn_conv_kernels = readString(is, "checkpoint cnn_conv_kernels");
    metadata.cnn_conv_strides = readString(is, "checkpoint cnn_conv_strides");
    metadata.cnn_conv_paddings = readString(is, "checkpoint cnn_conv_paddings");
    metadata.cnn_pool_after = readString(is, "checkpoint cnn_pool_after");
    metadata.cnn_pool_kernels = readString(is, "checkpoint cnn_pool_kernels");
    metadata.cnn_pool_strides = readString(is, "checkpoint cnn_pool_strides");
    metadata.cnn_fc_hidden_sizes = readString(is, "checkpoint cnn_fc_hidden_sizes");
    metadata.conv_backend = readString(is, "checkpoint conv_backend");

    if (metadata.batch_size <= 0) {
        throw std::runtime_error("Invalid checkpoint batch size.");
    }
    if (metadata.learning_rate <= 0.0) {
        throw std::runtime_error("Invalid checkpoint learning rate.");
    }
    return metadata;
}

std::ifstream openInput(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("Failed to open checkpoint for reading: " + path);
    }
    return is;
}

std::ofstream openOutput(const std::string& path) {
    const fs::path checkpoint_path(path);
    if (!checkpoint_path.parent_path().empty()) {
        fs::create_directories(checkpoint_path.parent_path());
    }

    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) {
        throw std::runtime_error("Failed to open checkpoint for writing: " + path);
    }
    return os;
}

void readAndValidateHeader(std::istream& is) {
    char magic[sizeof(kCheckpointMagic) - 1] = {};
    is.read(magic, sizeof(magic));
    if (!is) {
        throw std::runtime_error("Failed to read checkpoint header.");
    }
    if (std::string(magic, sizeof(magic)) != std::string(kCheckpointMagic, sizeof(kCheckpointMagic) - 1)) {
        throw std::runtime_error("Invalid checkpoint magic header.");
    }

    std::uint32_t version = 0;
    readExact(is, version, "checkpoint version");
    if (version != kCheckpointVersion) {
        throw std::runtime_error(
            "Unsupported checkpoint version: " + std::to_string(version) + ".");
    }
}

void writeHeader(std::ostream& os) {
    os.write(kCheckpointMagic, sizeof(kCheckpointMagic) - 1);
    if (!os) {
        throw std::runtime_error("Failed to write checkpoint header.");
    }
    writeExact(os, kCheckpointVersion, "checkpoint version");
}

}  // namespace

Metadata loadMetadata(const std::string& path) {
    std::ifstream is = openInput(path);
    readAndValidateHeader(is);
    return readMetadata(is);
}

void saveCheckpoint(const std::string& path,
                    const Metadata& metadata,
                    const std::vector<Node::Ptr>& parameters,
                    const Optimizer& optimizer) {
    std::ofstream os = openOutput(path);
    writeHeader(os);
    writeMetadata(os, metadata);

    const std::uint64_t param_count = static_cast<std::uint64_t>(parameters.size());
    writeExact(os, param_count, "checkpoint parameter count");
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        if (!parameters[i]) {
            throw std::runtime_error("Cannot save null parameter #" + std::to_string(i) + ".");
        }
        writeMatrix(os, parameters[i]->value(), "checkpoint parameter");
    }

    const std::string optimizer_type = optimizer.typeName();
    writeString(os, optimizer_type, "checkpoint optimizer type");
    optimizer.saveState(os);
}

Metadata loadCheckpoint(const std::string& path,
                        const std::vector<Node::Ptr>& parameters,
                        Optimizer& optimizer) {
    std::ifstream is = openInput(path);
    readAndValidateHeader(is);
    Metadata metadata = readMetadata(is);

    std::uint64_t saved_param_count = 0;
    readExact(is, saved_param_count, "checkpoint parameter count");
    if (saved_param_count != parameters.size()) {
        throw std::runtime_error(
            "Checkpoint parameter count mismatch: expected " +
            std::to_string(parameters.size()) + ", got " +
            std::to_string(saved_param_count) + ".");
    }

    for (std::size_t i = 0; i < parameters.size(); ++i) {
        if (!parameters[i]) {
            throw std::runtime_error("Cannot load into null parameter #" + std::to_string(i) + ".");
        }
        Matrix loaded = readMatrix(is, "checkpoint parameter");
        Matrix& current = const_cast<Matrix&>(parameters[i]->value());
        if (loaded.rows != current.rows || loaded.cols != current.cols) {
            throw std::runtime_error(
                "Checkpoint parameter shape mismatch at index " + std::to_string(i) + ".");
        }
        current = std::move(loaded);
        parameters[i]->zeroGrad();
    }

    const std::string optimizer_type = readString(is, "checkpoint optimizer type");
    if (optimizer_type != optimizer.typeName()) {
        throw std::runtime_error(
            "Checkpoint optimizer mismatch: expected '" + optimizer.typeName() +
            "', got '" + optimizer_type + "'.");
    }
    optimizer.loadState(is);
    return metadata;
}

}  // namespace checkpoint
