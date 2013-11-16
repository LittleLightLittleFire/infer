#include "util.h"
#include "bp.h"
#include "qp.h"
#include "mst.h"
#include "trbp.h"

#include "cuda/bp.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "lodepng.h"

namespace {
    const unsigned max_iter = 50;
    const unsigned samples = 200;
    const bool sync = false;

    const float lambda = 20;
    const float smooth_trunc = 2;
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cout << "usage ./stero [labels] [scale] [left.png] [right.png] [output.png]" << std::endl;
        return 1;
    }

    const unsigned labels = atoi(argv[1]);
    const unsigned scale = atoi(argv[2]);
    const char *left_name = argv[3];
    const char *right_name = argv[4];
    const char *output_name = argv[5];

    std::vector<unsigned char> left, right;
    unsigned width, height;

    { // image loading
        std::vector<unsigned char> left_rgba, right_rgba;

        if (lodepng::decode(left_rgba, width, height, left_name) || lodepng::decode(right_rgba, width, height, right_name)) {
            std::cout << "error loading images" << std::endl;
            return 1;
        }

        for (unsigned i = 0; i < left_rgba.size(); i += 4) {
            // convert to greyscale using hdtv conversion parameters
            left.push_back(0.2989 * left_rgba[i] + 0.5870 * left_rgba[i + 1] + 0.1140 * left_rgba[i + 2]);
            right.push_back(0.2989 * right_rgba[i] + 0.5870 * right_rgba[i + 1] + 0.1140 * right_rgba[i + 2]);
        }
    }

    const infer::indexer idx(width, height);
    const infer::indexer ndx(width, height, labels);

    // a 2d array of [index, label] represented as a flat array
    std::vector<float> unary(width * height * labels);

    { // create the potentials using a window size of 1
        for (unsigned y = 0; y < height; ++y) {
            for (unsigned x = labels - 1; x < width; ++x) { // offset the index so we don't go out of bounds
                const unsigned index = idx(x, y);
                for (unsigned p = 0; p < labels; ++p) {
                    unary[ndx(x, y) + p] = std::abs(static_cast<float>(left[index]) - static_cast<float>(right[index - p]));
                }
            }
        }
    }

    /*
    // create the grid CRF with the specified size
    infer::crf crf(width, height, labels, unary, lambda, 1, smooth_trunc);
    //infer::bp method(crf, sync);
    //infer::qp method(crf);
    infer::trbp method(crf, infer::sample_edge_apparence(width, height, samples), sync);
    //infer::trbp method(crf, std::vector<float>(width * height * 2, 1), sync);
    */
    infer::cuda::crf crf(width, height, labels, unary, lambda, 1, smooth_trunc);
    infer::cuda::bp method(crf);

    {
        const float unary_energy = method.unary_energy();
        const float pairwise_energy = method.pairwise_energy();
        std::cout << "initial" << " " << unary_energy + pairwise_energy << " " << unary_energy << " " << pairwise_energy << std::endl;
    }

    // run for 10 iterations
    for (unsigned i = 0; i < max_iter; ++i) {
        method.run(1);
        const float unary_energy = method.unary_energy();
        const float pairwise_energy = method.pairwise_energy();
        std::cout << i << " " << unary_energy + pairwise_energy << " " << unary_energy << " " << pairwise_energy << std::endl;
    }

    // convert the results into an image
    std::vector<unsigned char> image(width * height * 4);
    std::vector<unsigned> result = method.get_result();

    for (unsigned i = 0; i < width * height; ++i) {
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
