#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib>

#include "lodepng.h"

namespace {
    typedef unsigned int uint;
    typedef unsigned char uchar;

    const float linear_scaling = 0.07;
    const float data_trunc = 15.0;
    const float data_ceiling = 1.7;

    /** Makes indexing two dimensional arrays a bit easier */
    struct indexer {
        const uint width_, height_;
        explicit indexer(const uint width, const uint height) : width_(width), height_(height) { }

        /** returns the index in a 1d array */
        uint operator()(const uint x, const uint y) const {
            return x + width_ * y;
        }

        template <typename T>
        const T &operator()(const std::vector<T> &vec, const uint x, const uint y) const {
            return vec[operator()(x, y)];
        }

        template <typename T>
        T &operator()(std::vector<T> &vec, const uint x, const uint y) const {
            return vec[operator()(x, y)]; // too lazy to do the const cast trick
        }

    };

    /** A vector of floats */
    template <uint labels>
    struct message {
        float values[labels];
        explicit message() : values() { } // default initalisation to 0
    };

    /** Compute a new message from inputs */
    template <uint labels>
    void inline send_msg(const message<labels> &m1, const message<labels> &m2, const message<labels> &m3, const message<labels> &pot, message<labels> &out) {
        // compute the new message partially, deal with the pairwise term later
        for (uint i = 0; i < labels; ++i) {
            out.values[i] = m1.values[i] + m2.values[i] + m3.values[i] + pot.values[i]; // expontential form so multiplication is now addition
        }

        { // truncate
            // normally we would be forced to multiply this into the pairwise term which is has the size: labels^2, that would be painful
            // but since the pairwise term in a special form - max(y_i - y_j, d) (truncated linear model), we could get the result in linear time
            // algorithm from Pedro F. Felzenszwalb and Daniel P. Huttenlocher (2006): Efficient Belief Propagation for Early Vision

            // compute the minimum to truncate with
            const float trunc = data_ceiling + *std::min_element(out.values, out.values + labels);

            // first pass, equivalent to a haskell `scanl (min . succ)`
            for (uint i = 1; i < labels; ++i) {
                out.values[i] = std::min(out.values[i - 1] + 1.0f, out.values[i]);
            }

            // second pass, same thing but with the list reversed
            for (int i = labels - 2; i >= 0; --i) {
                out.values[i] = std::min(out.values[i + 1] + 1.0f, out.values[i]);
            }

            std::transform(out.values, out.values + labels, out.values, [trunc](const float x){ return std::min(x, trunc); });
        }

        // normalise
        const float sum = std::accumulate(out.values, out.values + labels, 0.0f) / labels;
        std::transform(out.values, out.values + labels, out.values, [sum](const float x){ return x - sum; });
    }

    template <uint labels>
    std::vector<uchar> decode(const uint max_iter, const uint width, const uint height, const std::vector<message<labels>> &pot) {
        // references to [Waineright et al] (2005) : MAP Estimation Via agreement on Trees: Message-Passing and Linear Programming
        // todo: find the edge apparence probability: rho
        // create minimal spanning trees with the edges having random weights [0,1], until all the edges are covered, count edge apparences

        // allocate space for messages, in four directions (up, down, left and right)
        const uint nodes = width * height;
        std::vector<message<labels>> u(nodes), d(nodes), l(nodes), r(nodes);

        indexer idx(width, height);

        for (uint i = 0; i < max_iter; ++i) {
            std::cout << "iter: " << i << std::endl;
            // checkerboard update scheme
            for (uint y = 1; y < height - 1; ++y) {
                for (uint x = ((y + i) % 2) + 1; x < width - 1; x += 2) {
                    // send messages in each direction
                    send_msg(idx(u, x, y+1), idx(l, x+1, y), idx(r, x-1, y), idx(pot, x, y), idx(u, x, y));
                    send_msg(idx(d, x, y-1), idx(l, x+1, y), idx(r, x-1, y), idx(pot, x, y), idx(d, x, y));
                    send_msg(idx(u, x, y+1), idx(d, x, y-1), idx(r, x-1, y), idx(pot, x, y), idx(r, x, y));
                    send_msg(idx(u, x, y+1), idx(d, x, y-1), idx(l, x+1, y), idx(pot, x, y), idx(l, x, y));
                }
            }
        }

        // for each pixel: find the most likely label
        std::vector<uchar> result(nodes);
        for (uint y = 1; y < height - 1; ++y) {
            for (uint x = 1; x < width - 1; ++x) {
                const uint index = idx(x, y);
                uint min_label = 0;
                float min_value = std::numeric_limits<float>::max();

                for (uint i = 0; i < labels; ++i) {
                    const float val = u[idx(x, y+1)].values[i] + d[idx(x, y-1)].values[i] + l[idx(x+1, y)].values[i] + r[idx(x-1, y)].values[i] + pot[idx(x, y)].values[i];

                    if (val < min_value) {
                        min_label = i;
                        min_value = val;
                    }
                }

                result[index] = min_label;
            }
        }

        return result;
    }

}

int main(int argc, char *argv[]) {
    // constants initalisation
    const uint labels = 16;

    if (argc != 3) {
        std::cout << "usage ./stero [left.png] [right.png]" << std::endl;
        return 1;
    }

    std::vector<uchar> left, right;
    uint width, height;

    { // image loading
        std::vector<uchar> left_rgba, right_rgba;

        if (lodepng::decode(left_rgba, width, height, argv[1]) || lodepng::decode(right_rgba, width, height, argv[2])) {
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
    indexer idx(width, height);

    // compute the unary potential functions for each random variable (y_1 ... y_n)
    // using the sum of absolute differences

    // a 2d array of [index, label] represented as a flat array
    std::vector<message<labels>> unary_psi(nodes);

    { // create the potentials using a window size of 1
        for (uint y = 0; y < height; ++y) {
            for (uint x = labels - 1; x < width; ++x) { // offset the index so we don't go out of bounds
                const uint index = idx(x, y);
                for (uint p = 0; p < labels; ++p) {
                    unary_psi[index].values[p] = linear_scaling * std::min<float>(abs(static_cast<int>(left[index]) - right[index - p]), data_trunc);
                }
            }
        }
    }

    std::vector<uchar> result = decode<labels>(5, width, height, unary_psi);

    // convert the results into an image
    std::vector<uchar> image(result.size() * 4);
    for (uint i = 0; i < result.size(); ++i) {
        const float val = result[i] * (256.0f / labels);
        image[i * 4] = image[i * 4 + 1] = image[i * 4 + 2] = val;
        image[i * 4 + 3] = 255; // alpha channel
    }

    if (lodepng::encode("output.png", image, width, height)) {
        std::cout << "error writing image" << std::endl;
        return 2;
    }

    return 0;
}
