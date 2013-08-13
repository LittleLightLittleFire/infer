#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdlib>

#include "lodepng.h"

namespace {
    typedef unsigned int uint;
    typedef unsigned char uchar;
}

int main(int argc, char *argv[]) {
    // constants initalisation
    const uint labels = 16;
    const uint max_iter = 10;

    if (argc != 3) {
        std::cout << "usage ./stero [left.png] [right.png]" << std::endl;
        return 1;
    }

    std::vector<unsigned char> left;
    std::vector<unsigned char> right;

    uint width;
    uint height;

    // read the pngs
    lodepng::decode(left, width, height, argv[1]);
    lodepng::decode(right, width, height, argv[2]);

    // underlying model is of a grid
    const uint nodes = width * height;
    const uint edges = nodes * 2;

    // compute the unary potential functions for each random variable (y_1 ... y_n)
    // using the sum of absolute differences

    // a 2d array of [index, label] represented as a flat array
    std::vector<uchar> unary_psi(nodes * labels);

    {
        // we are using a window size of 1 pixel
        for (uint y = 0; y < height; ++y) {
            for (uint x = labels; x < width; ++x) { // offset the index so we don't go out of bounds
                for (uint p = 0; p < labels; ++p) {
                    const uint index = x + y * width;
                    unary_psi[index * labels + p] = static_cast<uchar>(abs(static_cast<int>(left[index]) - right[index - p]));
                }
            }
        }
    }

    // compute the edgewise term
    // using the discontinutity preserving prior
    // psi(y_i, y_j) = min { |y_i - y_j|, d_max }

    // references to [Waineright et al] (2005) : MAP Estimation Via agreement on Trees: Message-Passing and Linear Programming

    // find the edge probability rho [fig 1]
    // create minimal spanning trees with the edges having random weights [0,1], until all the edges are covered, count edge apparences

    std::vector<float> rho(edges);

    {
        std::vector<uint> count(edges);

        std::vector<float> weights(edges);
        std::vector<bool> tree(edges * edges);

        uint iterations = 0;
        while (std::any_of(count.cbegin(), count.cend(), [](const uint c) { return c == 0; } )) {
            ++iterations;

            // randomly assign weights
            std::transform(weights.begin(), weights.end(), weights.begin(), [](const uint c) { return rand() / static_cast<float>(RAND_MAX); });
        }

    }

    return 0;
}
