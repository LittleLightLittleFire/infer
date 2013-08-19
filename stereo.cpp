#include <iterator>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib>

#include "util.h"
#include "mst.h"
#include "lodepng.h"

namespace {
    typedef unsigned int uint;
    typedef unsigned char uchar;

    const float linear_scaling = 0.075;
    const float data_trunc = 15.0;
    const float data_disc = 1.7;

    /** Compute a new message from inputs */
    void inline send_msg(const float *const m1, const float *const m2, const float *const m3, const float *const opp, float *const out, const float *const pot, const float rm1, const float rm2, const float rm3, const float ropp, const uint labels) {
        // compute the new message partially, deal with the pairwise term later
        for (uint i = 0; i < labels; ++i) {
            out[i] = m1[i] *  rm1 + m2[i] * rm2 + m3[i] * rm3 + opp[i] * (ropp - 1) + pot[i];
        }

        { // truncate
            // normally we would be forced to multiply this into the pairwise term which is has the size: labels^2, that would be painful
            // but since the pairwise term in a special form - min(y_i - y_j, d) (truncated linear model), we could get the result in linear time
            // algorithm from Pedro F. Felzenszwalb and Daniel P. Huttenlocher (2006): Efficient Belief Propagation for Early Vision

            // compute the minimum to truncate with
            const float trunc = data_disc + *std::min_element(out, out + labels);

            // first pass, equivalent to a haskell `scanl (min . succ . (+ c))`
            for (uint i = 1; i < labels; ++i) {
                out[i] = std::min(out[i - 1] + (1 / ropp), out[i]);
            }

            // second pass, same thing but with the list reversed
            for (int i = labels - 2; i >= 0; --i) {
                out[i] = std::min(out[i + 1] + (1 / ropp), out[i]);
            }

            std::transform(out, out + labels, out, [trunc](const float x){ return std::min(x, trunc); });
        }

        // normalise
        const float sum = std::accumulate(out, out + labels, 0.0f) / labels;
        std::transform(out, out + labels, out, [sum](const float x){ return x - sum; });
    }

    std::vector<uchar> decode(const uint labels, const uint max_iter, const uint width, const uint height, const std::vector<float> &pot, const std::vector<float> &rho) {
        // allocate space for messages, in four directions (up, down, left and right)
        const uint nodes = width * height;
        const uint elements = labels * nodes;

        const edge_indexer edx(width, height);
        const indexer ndx(width, height, labels);

        // convenience functions
        const auto left = move::LEFT, right = move::RIGHT, up = move::UP, down = move::DOWN;
        const auto get = [&rho, &edx](const move m, const uint x, const uint y) {
            return rho[edx(x, y, m)];
        };

        std::vector<float> u(elements), d(elements), l(elements), r(elements);
        for (uint i = 0; i < max_iter; ++i) {
            std::cout << "iter: " << i << std::endl;
            // checkerboard update scheme
            for (uint y = 1; y < height - 1; ++y) {
                for (uint x = ((y + i) % 2) + 1; x < width - 1; x += 2) {

                    // send messages in each direction
                    //        m1                   m2                   m3                   opp                     out             pot
                    send_msg(ndx(u,  x, y+1),    ndx(l   , x+1, y),  ndx(r,     x-1, y),  ndx(d,    x, y-1),    ndx(u, x, y),    ndx(pot, x, y),
                             get(up, x, y+1),    get(left, x+1, y),  get(right, x-1, y),  get(down, x, y-1), labels);

                    send_msg(ndx(d,    x, y-1),  ndx(l   , x+1, y),  ndx(r,     x-1, y),  ndx(u,  x, y+1),      ndx(d, x, y),    ndx(pot, x, y),
                             get(down, x, y-1),  get(left, x+1, y),  get(right, x-1, y),  get(up, x, y+1), labels);

                    send_msg(ndx(u,  x, y+1),    ndx(d,    x, y-1),  ndx(r,     x-1, y),  ndx(l,    x+1, y),    ndx(r, x, y),    ndx(pot, x, y),
                             get(up, x, y+1),    get(down, x, y-1),  get(right, x-1, y),  get(left, x+1, y), labels);

                    send_msg(ndx(u,  x, y+1),    ndx(d,    x, y-1),  ndx(l,    x+1, y),   ndx(r,     x-1, y),   ndx(l, x, y),    ndx(pot, x, y),
                             get(up, x, y+1),    get(down, x, y-1),  get(left, x+1, y),   get(right, x-1, y), labels);
                }
            }
        }

        std::vector<uchar> result(nodes);
        const indexer idx(width, height);

        // for each pixel: find the most likely label
        for (uint y = 1; y < height - 1; ++y) {
            for (uint x = 1; x < width - 1; ++x) {
                uint min_label = 0;
                float min_value = std::numeric_limits<float>::max();

                for (uint i = 0; i < labels; ++i) {
                    const float val = u[ndx(x, y+1) + i] * get(up,    x, y+1)
                                    + d[ndx(x, y-1) + i] * get(down,  x, y-1)
                                    + l[ndx(x+1, y) + i] * get(left,  x+1, y)
                                    + r[ndx(x-1, y) + i] * get(right, x-1, y)
                                    + pot[ndx(x, y) + i];

                    if (val < min_value) {
                        min_label = i;
                        min_value = val;
                    }
                }

                result[idx(x, y)] = min_label;
            }
        }

        return result;
    }

}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cout << "usage ./stero [labels] [left.png] [right.png] [output.png]" << std::endl;
        return 1;
    }

    // constants initalisation
    const uint mst_samples = 50;

    const uint labels = atoi(argv[1]);
    const char *left_name = argv[2];
    const char *right_name = argv[3];
    const char *output_name = argv[4];

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
    std::vector<uchar> result = decode(labels, 200, width, height, unary_psi, rho);

    // convert the results into an image
    std::vector<uchar> image(result.size() * 4);
    for (uint i = 0; i < result.size(); ++i) {
        const float val = result[i] * (256.0f / labels);
        image[i * 4] = image[i * 4 + 1] = image[i * 4 + 2] = val;
        image[i * 4 + 3] = 255; // alpha channel
    }

    if (lodepng::encode(output_name, image, width, height)) {
        std::cout << "error writing image" << std::endl;
        return 2;
    }

    return 0;
}
