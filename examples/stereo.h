#ifndef STEREO_H
#define STEREO_H

#include "crf.h"
#include "util.h"

#include "lodepng.h"
#include <functional>
#include <string>
#include <stdexcept>
#include <cmath>

namespace stereo {
    const float lambda = 20;
    const float smooth_trunc = 2;

    void run(const std::function<const std::vector<unsigned>(const infer::crf)> method, const unsigned labels, const unsigned scale, const std::string left_name, const std::string right_name, const std::string output_name) {
        std::vector<unsigned char> left, right;
        unsigned width, height;

        { // load the image
            std::vector<unsigned char> left_rgba, right_rgba;

            if (lodepng::decode(left_rgba, width, height, left_name) || lodepng::decode(right_rgba, width, height, right_name)) {
                throw std::runtime_error("Error loading images");
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
                        unary[ndx(x, y, p)] = std::abs(static_cast<float>(left[index]) - static_cast<float>(right[index - p]));
                    }
                }
            }
        }

        std::vector<unsigned> result = method(infer::crf(width, height, labels, unary, lambda, 1, smooth_trunc));
        std::vector<unsigned char> image(result.size() * 4);
        for (unsigned i = 0; i < result.size(); ++i) {
            const float val = result[i] * scale;
            image[i * 4] = image[i * 4 + 1] = image[i * 4 + 2] = val;
            image[i * 4 + 3] = 255; // alpha channel
        }

        if (lodepng::encode(output_name, image, width, height)) {
            throw std::runtime_error("Error writing image");
        }
    }
}

#endif // STEREO_H
