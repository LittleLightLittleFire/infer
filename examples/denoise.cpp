#include "driver.h"

#include "crf.h"
#include "lodepng.h"

#include <iostream>
#include <string>
#include <cmath>
#include <stdexcept>

namespace denoise {

const unsigned labels = 256;

const float trunc = 200;
const float lambda = 25;

void run(const std::function<const std::vector<unsigned>(const infer::crf)> method, const std::string image_name, const std::string output_name) {
    unsigned width, height;

    std::vector<float> unary;
    std::vector<float> pairwise;

    { // load the image and create potentials
        std::vector<unsigned char> image_rgba;

        if (lodepng::decode(image_rgba, width, height, image_name)) {
            throw std::runtime_error("Error loading images");
        }

        for (unsigned i = 0; i < image_rgba.size(); i += 4) {
            const unsigned intensity = image_rgba[i];

            for (unsigned j = 0; j < labels; ++j) {
                const float diff = static_cast<float>(j) - static_cast<float>(intensity);
                unary.push_back(diff * diff);
            }
        }

        // just make the L2 quadratic potentials
        for (unsigned i = 0; i < labels; ++i) {
            for (unsigned j = 0; j < labels; ++j) {
                const float diff = static_cast<float>(i) - static_cast<float>(j);
                pairwise.push_back(std::min(diff * diff, trunc));
            }
        }

    }

    std::vector<unsigned> result = method(infer::crf(width, height, labels, unary, lambda, true, pairwise));

    std::vector<unsigned char> result_argb;
    for (unsigned i = 0; i < result.size(); ++i) {
        result_argb.push_back(result[i]);
        result_argb.push_back(result[i]);
        result_argb.push_back(result[i]);
        result_argb.push_back(255);
    }

    if (lodepng::encode(output_name, result_argb, width, height)) {
        throw std::runtime_error("Error writing image");
    }
}

}
