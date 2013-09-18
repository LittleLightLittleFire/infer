#include <iterator>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib>

#include "mst.h"
#include "util.h"
#include "trhbp.h"

#include "lodepng.h"

namespace {
    typedef unsigned int uint;
    typedef unsigned char uchar;

    // constants initalisation
    const float lambda = 20;
    const float smooth_trunc = 2;

    const uint mst_samples = 200;
    const bool lbp = false;
    const bool sync = true;

    //const uint layer_spec[] = { 5, 5, 5, 5, 5 };
    const uint layer_spec[] = { 200 };
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cout << "usage ./stero [labels] [scale] [left.png] [right.png] [output.png]" << std::endl;
        return 1;
    }

    const uint labels = atoi(argv[1]);
    const uint scale = atoi(argv[2]);
    const char *left_name = argv[3];
    const char *right_name = argv[4];
    const char *output_name = argv[5];

    std::vector<uchar> left, right;
    uint width, height;

    { // image loading
        std::vector<uchar> left_rgba, right_rgba;

        if (lodepng::decode(left_rgba, width, height, left_name) || lodepng::decode(right_rgba, width, height, right_name)) {
            std::cout << "error loading images" << std::endl;
            return 1;
        }

        for (uint i = 0; i < left_rgba.size(); i += 4) {
            // convert to greyscale using hdtv conversion parameters
            left.push_back(0.2989 * left_rgba[i] + 0.5870 * left_rgba[i + 1] + 0.1140 * left_rgba[i + 2]);
            right.push_back(0.2989 * right_rgba[i] + 0.5870 * right_rgba[i + 1] + 0.1140 * right_rgba[i + 2]);
        }
    }

    // underlying model is of a grid
    const uint nodes = width * height;

    const indexer idx(width, height);
    const indexer ndx(width, height, labels);

    // compute the unary potential functions for each random variable (y_1 ... y_n)
    // using the sum of absolute differences

    // a 2d array of [index, label] represented as a flat array
    std::vector<float> unary_psi(nodes * labels);

    { // create the potentials using a window size of 1
        for (uint y = 0; y < height; ++y) {
            for (uint x = labels - 1; x < width; ++x) { // offset the index so we don't go out of bounds
                const uint index = idx(x, y);
                for (uint p = 0; p < labels; ++p) {
                    unary_psi[ndx(x, y) + p] = std::abs(float(left[index]) - float(right[index - p]));
                }
            }
        }
    }

    std::vector<uint> layers(layer_spec, layer_spec + sizeof(layer_spec) / sizeof(layer_spec[0]));
    std::vector<std::vector<float>> rho;
    { // create the edge apparence probabilities
        uint w = width;
        uint h = height;

        for (uint i = 0; i < layers.size(); ++i) {
            rho.push_back(lbp ? std::vector<float>(width * height * 2, 1) : sample_edge_apparence(w, h, mst_samples));

            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
    }

    std::vector<uchar> result = decode_trhbp(labels, layers, width, height, unary_psi, rho, lambda, smooth_trunc, sync);

    // convert the results into an image
    std::vector<uchar> image(result.size() * 4);
    for (uint i = 0; i < result.size(); ++i) {
        const float val = result[i] * scale;
        image[i * 4] = image[i * 4 + 1] = image[i * 4 + 2] = val;
        image[i * 4 + 3] = 255; // alpha channel
    }

    if (lodepng::encode(output_name, image, width, height)) {
        std::cout << "error writing image" << std::endl;
        return 2;
    }

    return 0;
}
