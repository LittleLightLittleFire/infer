#include <iterator>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>
#include <limits>

#include "util.h"
#include "mst.h"
#include "lodepng.h"

namespace {
    typedef unsigned int uint;
    typedef unsigned char uchar;

    const float linear_scaling = 0.07;
    const float data_trunc = 15.0;
    const float data_ceiling = 1.7;

    /** A vector of floats */
    template <uint labels>
    struct message {
        float values[labels];
        explicit message() : values() { } // default initalisation to 0
    };

    float edgewise(const float x, const float y) {
        return std::min(std::abs(x - y), data_ceiling);
    }

    /** Compute a new message from inputs */
    template <uint labels>
    void inline send_msg(const message<labels> &m1, const message<labels> &m2, const message<labels> &m3, const message<labels> &opp, message<labels> &out, const message<labels> &pot, const float rm1, const float rm2, const float rm3, const float ropp) {
        // compute the new message partially, deal with the pairwise term later
        for (uint i = 0; i < labels; ++i) {
            out.values[i] = m1.values[i] *  rm1 + m2.values[i] * rm2 + m3.values[i] * rm3 + opp.values[i] * (ropp - 1) + pot.values[i];
        }

        // find the edgewise with the minimal potential
        for (uint i = 0; i < labels; ++i) {
            float min_value = std::numeric_limits<float>::max();
            for (uint j = 0; j < labels; ++j) {
                const float value = out.values[i] + edgewise(i, j) * (1 / ropp);
                if (value < min_value) {
                    min_value = value;
                }
            }

            out.values[i] = min_value;
        }

        // normalise
        const float sum = std::accumulate(out.values, out.values + labels, 0.0f) / labels;
        std::transform(out.values, out.values + labels, out.values, [sum](const float x){ return x - sum; });
    }

    template <uint labels>
    std::vector<uchar> decode(const uint max_iter, const uint width, const uint height, const std::vector<message<labels>> &pot, const std::vector<float> &rho) {
        // allocate space for messages, in four directions (up, down, left and right)
        const uint nodes = width * height;
        std::vector<message<labels>> u(nodes), d(nodes), l(nodes), r(nodes);

        indexer idx(width, height);
        edge_indexer edx(width, height);

        // convience functions
        const auto left = move::LEFT, right = move::RIGHT, up = move::UP, down = move::DOWN;
        const auto get = [&rho, &edx](const move m, const uint x, const uint y) {
            return rho[edx(x, y, m)];
        };

        for (uint i = 0; i < max_iter; ++i) {
            std::cout << "iter: " << i << std::endl;
            // checkerboard update scheme
            for (uint y = 1; y < height - 1; ++y) {
                for (uint x = ((y + i) % 2) + 1; x < width - 1; x += 2) {

                    // send messages in each direction
                    //        m1                   m2                   m3                   opp                     out             pot
                    send_msg(idx(u,  x, y+1),    idx(l   , x+1, y),  idx(r,     x-1, y),  idx(d,    x, y-1),    idx(u, x, y),    idx(pot, x, y),
                             get(up, x, y+1),    get(left, x+1, y),  get(right, x-1, y),  get(down, x, y-1));

                    send_msg(idx(d,    x, y-1),  idx(l   , x+1, y),  idx(r,     x-1, y),  idx(u,  x, y+1),      idx(d, x, y),    idx(pot, x, y),
                             get(down, x, y-1),  get(left, x+1, y),  get(right, x-1, y),  get(up, x, y+1));

                    send_msg(idx(u,  x, y+1),    idx(d,    x, y-1),  idx(r,     x-1, y),  idx(l,    x+1, y),    idx(r, x, y),    idx(pot, x, y),
                             get(up, x, y+1),    get(down, x, y-1),  get(right, x-1, y),  get(left, x+1, y));

                    send_msg(idx(u,  x, y+1),    idx(d,    x, y-1),  idx(l,    x+1, y),   idx(r,     x-1, y),   idx(l, x, y),    idx(pot, x, y),
                             get(up, x, y+1),    get(down, x, y-1),  get(left, x+1, y),   get(right, x-1, y));
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
                    const float val = u[idx(x, y+1)].values[i] * get(up,    x, y+1)
                                    + d[idx(x, y-1)].values[i] * get(down,  x, y-1)
                                    + l[idx(x+1, y)].values[i] * get(left,  x+1, y)
                                    + r[idx(x-1, y)].values[i] * get(right, x-1, y)
                                    + pot[idx(x, y)].values[i];

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
    const uint mst_samples = 20;

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

    // sample the grid
    std::vector<uint> edge_samples = sample_edge_apparence(width, height, mst_samples);

    std::vector<float> rho;
    std::transform(edge_samples.begin(), edge_samples.end(), std::back_inserter(rho), [](const uchar count){ return static_cast<float>(count) / mst_samples; });

    std::cout << "finished running " << mst_samples << " samples" << std::endl;
    std::vector<uchar> result = decode<labels>(5, width, height, unary_psi, rho);

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
