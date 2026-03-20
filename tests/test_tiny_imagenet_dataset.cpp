#include "tiny_imagenet_dataset.hpp"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <jpeglib.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void writeJpeg(const fs::path& path,
               int width,
               int height,
               const std::vector<unsigned char>& rgb) {
    FILE* out = std::fopen(path.string().c_str(), "wb");
    if (!out) {
        throw std::runtime_error("Could not open output JPEG: " + path.string());
    }

    jpeg_compress_struct cinfo{};
    jpeg_error_mgr jerr{};
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, out);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 100, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row =
            const_cast<JSAMPROW>(&rgb[static_cast<std::size_t>(cinfo.next_scanline) *
                                      static_cast<std::size_t>(width) * 3]);
        jpeg_write_scanlines(&cinfo, &row, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    std::fclose(out);
}

std::vector<unsigned char> solidRgb(int width, int height,
                                    unsigned char r,
                                    unsigned char g,
                                    unsigned char b) {
    std::vector<unsigned char> rgb(static_cast<std::size_t>(width) *
                                   static_cast<std::size_t>(height) * 3);
    for (std::size_t i = 0; i < rgb.size(); i += 3) {
        rgb[i] = r;
        rgb[i + 1] = g;
        rgb[i + 2] = b;
    }
    return rgb;
}

void writeText(const fs::path& path, const std::string& text) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not write file: " + path.string());
    }
    out << text;
}

} // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "tiny_imagenet_dataset_test";
    const fs::path nested_root = root / "tiny-imagenet-200";
    fs::remove_all(root);

    try {
        fs::create_directories(nested_root / "train" / "n00000001" / "images");
        fs::create_directories(nested_root / "train" / "n00000002" / "images");
        fs::create_directories(nested_root / "val" / "images");

        writeText(nested_root / "wnids.txt", "n00000001\nn00000002\n");
        writeText(nested_root / "val" / "val_annotations.txt",
                  "val_0.JPEG\tn00000001\t0\t0\t64\t64\n"
                  "val_1.JPEG\tn00000002\t0\t0\t64\t64\n");

        writeJpeg(nested_root / "train" / "n00000001" / "images" / "train_0.JPEG",
                  64, 64, solidRgb(64, 64, 255, 0, 0));
        writeJpeg(nested_root / "train" / "n00000002" / "images" / "train_1.JPEG",
                  64, 64, solidRgb(64, 64, 0, 255, 0));
        writeJpeg(nested_root / "val" / "images" / "val_0.JPEG",
                  64, 64, solidRgb(64, 64, 0, 0, 255));
        writeJpeg(nested_root / "val" / "images" / "val_1.JPEG",
                  64, 64, solidRgb(64, 64, 255, 255, 0));

        TinyImageNetDataset dataset(root.string());
        dataset.load();

        assert(dataset.info().name == "tiny-imagenet");
        assert(dataset.info().input_channels == 3);
        assert(dataset.info().input_height == 64);
        assert(dataset.info().input_width == 64);
        assert(dataset.info().input_dim == 64 * 64 * 3);
        assert(dataset.info().num_classes == 2);

        const Matrix& x_train = dataset.getTrainImages();
        const Matrix& y_train = dataset.getTrainLabels();
        const Matrix& x_val = dataset.getTestImages();
        const Matrix& y_val = dataset.getTestLabels();

        assert(x_train.rows == 2);
        assert(x_train.cols == 64 * 64 * 3);
        assert(y_train.rows == 2);
        assert(y_train.cols == 2);
        assert(x_val.rows == 2);
        assert(y_val.cols == 2);

        assert(y_train(0, 0) == 1.0 || y_train(0, 1) == 1.0);
        assert(y_train(1, 0) == 1.0 || y_train(1, 1) == 1.0);

        for (double v : x_train.data) {
            assert(v >= 0.0);
            assert(v <= 1.0);
        }
        for (double v : x_val.data) {
            assert(v >= 0.0);
            assert(v <= 1.0);
        }
    } catch (...) {
        fs::remove_all(root);
        throw;
    }

    fs::remove_all(root);
    return 0;
}
