#ifndef ISEG_H
#define ISEG_H

#include "crf.h"
#include "util.h"

#include "lodepng.h"
#include <functional>
#include <numeric>
#include <string>
#include <stdexcept>
#include <cmath>

namespace iseg {
    const float gamma = 10;
    const float lambda_1 = 5;
    const float lambda_2 = 50;

    void run(const std::function<const std::vector<unsigned>(const infer::crf)> method, const std::string image_name, const std::string anno_name, const std::string output_name) {
        unsigned width, height;

        std::vector<float> unary;
        std::vector<float> pairwise;

        // intensities of the image (greyscale)
        std::vector<unsigned char> image;

        std::vector<unsigned char> anno_rgba;

        { // load the image and create potentials
            std::vector<unsigned char> image_rgba;

            if (lodepng::decode(image_rgba, width, height, image_name) || lodepng::decode(anno_rgba, width, height, anno_name)) {
                throw std::runtime_error("Error loading images");
            }

            // histograms for foreground and background
            std::vector<unsigned> hist_fg(256);
            std::vector<unsigned> hist_bg(256);

            // represents the annotation, 0 unlabeled, 1 for fg, 2 for bg
            std::vector<unsigned char> affinity;

            for (unsigned i = 0; i < image_rgba.size(); i += 4) {
                const unsigned intensity = image_rgba[i];

                // generate histograms
                if (anno_rgba[i] == 255 && anno_rgba[i + 1] == 0 && anno_rgba[i+2] == 0) {
                    hist_fg[intensity] += 1;
                    affinity.push_back(1);
                } else if (anno_rgba[i] == 0 && anno_rgba[i + 1] == 0 && anno_rgba[i+2] == 255) {
                    hist_bg[intensity] += 1;
                    affinity.push_back(2);
                } else {
                    affinity.push_back(0);
                }

                image.push_back(intensity);
            }

            // generate summations for normalisation
            const double hist_fg_sum = std::accumulate(hist_fg.cbegin(), hist_fg.cend(), 0.0);
            const double hist_bg_sum = std::accumulate(hist_bg.cbegin(), hist_bg.cend(), 0.0);

            for (unsigned i = 0; i < image.size(); ++i) {
                const unsigned intensity = image[i];

                const float bonus = affinity[i] != 0 ? gamma : 0;

                // assign unary potentials
                unary.push_back(- std::log(hist_fg[intensity] / hist_fg_sum) - std::log(1 - (hist_bg[intensity] / hist_bg_sum)) + bonus);
                unary.push_back(- std::log(1 - (hist_fg[intensity] / hist_fg_sum)) - std::log(hist_bg[intensity] / hist_bg_sum) + bonus);

                // assign pairwise potentials for down and right edge
                unsigned x = i % width, y = i / width;

                auto make_pairwise = [&pairwise, &image, x, y, width, height](const unsigned x2, const unsigned y2) {
                    const infer::indexer idx(width, height);
                    // bound check
                    if (x2 > width - 1 || y2 > height - 1) {
                        for (unsigned j = 0; j < 4; ++j) {
                            pairwise.push_back(0);
                        }
                    } else {
                        // constrast dependent smoothness term
                        for (unsigned j = 0; j < 2; ++j) {
                            for (unsigned k = 0; k < 2; ++k) {
                                if (j != k) {
                                    const float diff = static_cast<float>(image[idx(x, y)]) - static_cast<float>(image[idx(x2, y2)]);
                                    pairwise.push_back(lambda_1 + lambda_2 * std::exp(0.5 * sqrt(std::abs(diff))));
                                } else {
                                    pairwise.push_back(0);
                                }
                            }
                        }
                    }
                };

                make_pairwise(x, y + 1);
                make_pairwise(x + 1, y);
            }
        }

        std::vector<unsigned> result = method(infer::crf(width, height, 2, unary, 1, pairwise));

        // overwrite on top of anno_rgba
        for (unsigned i = 0; i < result.size(); ++i) {
            if (result[i] == 0) { // bg
                anno_rgba[i + 2] = 255;
            } else { //fg
                anno_rgba[i] = 255;
            }
        }

        if (lodepng::encode(output_name, anno_rgba, width, height)) {
            throw std::runtime_error("Error writing image");
        }
    }
}

#endif // ISEG_H
