#include <iterator>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib>
#include <functional>

#include "util.h"
#include "mst.h"
#include "bp.h"

#include "lodepng.h"

namespace {
    typedef unsigned int uint;
    typedef unsigned char uchar;

    // constants initalisation
    const uint mst_samples = 50;

    const float linear_scaling = 0.075;
    const float data_disc = 1.7;
    const float data_trunc = 15;
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
                    unary_psi[ndx(x, y) + p] = linear_scaling * std::min<float>(abs(static_cast<int>(left[index]) - right[index - p]), data_trunc);
                }
            }
        }
    }

    // sample the grid
    std::vector<uint> edge_samples = sample_edge_apparence(width, height, mst_samples);

    std::vector<float> rho;
    std::transform(edge_samples.begin(), edge_samples.end(), std::back_inserter(rho), [](const uchar count){ return static_cast<float>(count) / mst_samples; });

    std::cout << "finished running " << mst_samples << " samples" << std::endl;
    std::vector<uchar> result = decode_trbp(labels, 200, width, height, unary_psi, rho, std::bind(send_msg_lin_trunc, std::placeholders::_1, data_disc), true);

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
