#include <algorithm>

#include "bp.h"
#include "util.h"

namespace {
    typedef unsigned int uint;
    typedef unsigned char uchar;
}

void send_msg_lin_trunc(const message_data in, float lambda, const float smooth_trunc) {
    // compute the new message partially, deal with the pairwise term later
    for (uint i = 0; i < in.labels; ++i) {
        in.out[i] = in.m1[i] * in.rm1 + in.m2[i] * in.rm2 + in.m3[i] * in.rm3 + in.opp[i] * (in.ropp - 1) + in.pot[i];
    }

    // because of trbp
    lambda *= (1 / in.ropp);

    // derive s and t
    const float s = lambda;
    const float t = lambda * smooth_trunc;

    { // calculate the pairwise term using the O(n) algorithm
        const float trunc = t + *std::min_element(in.out, in.out + in.labels);

        for (uint i = 1; i < in.labels; ++i) {
            in.out[i] = std::min(in.out[i - 1] + s, in.out[i]);
        }

        for (int i = in.labels - 2; i >= 0; --i) {
            in.out[i] = std::min(in.out[i + 1] + s, in.out[i]);
        }

        std::transform(in.out, in.out + in.labels, in.out, [trunc](const float x){ return std::min(x, trunc); });
    }

    // normalise
    const float sum = std::accumulate(in.out, in.out + in.labels, 0.0f) / in.labels;
    std::transform(in.out, in.out + in.labels, in.out, [sum](const float x){ return x - sum; });
}

std::vector<uchar> decode_trbp(const uint labels, const uint max_iter, const uint width, const uint height, const std::vector<float> &pot, const std::vector<float> &rho, const std::function<void(message_data)> send_msg, const bool sync) {
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
        if (sync) {
            // checkerboard update scheme
            for (uint y = 1; y < height - 1; ++y) {
                for (uint x = ((y + i) % 2) + 1; x < width - 1; x += 2) {
                    // send messages in each direction
                    //        m1                   m2                   m3                   opp                     pot             out
                    send_msg({ ndx(u,  x, y+1),    ndx(l   , x+1, y),  ndx(r,     x-1, y),  ndx(d,    x, y-1),    ndx(pot, x, y),    ndx(u, x, y),
                             get(up, x, y+1),    get(left, x+1, y),  get(right, x-1, y),  get(down, x, y-1), labels });

                    send_msg({ ndx(d,    x, y-1),  ndx(l   , x+1, y),  ndx(r,     x-1, y),  ndx(u,  x, y+1),      ndx(pot, x, y),    ndx(d, x, y),
                             get(down, x, y-1),  get(left, x+1, y),  get(right, x-1, y),  get(up, x, y+1), labels });

                    send_msg({ ndx(u,  x, y+1),    ndx(d,    x, y-1),  ndx(r,     x-1, y),  ndx(l,    x+1, y),    ndx(pot, x, y),    ndx(r, x, y),
                             get(up, x, y+1),    get(down, x, y-1),  get(right, x-1, y),  get(left, x+1, y), labels });

                    send_msg({ ndx(u,  x, y+1),    ndx(d,    x, y-1),  ndx(l,    x+1, y),   ndx(r,     x-1, y),   ndx(pot, x, y),    ndx(l, x, y),
                             get(up, x, y+1),    get(down, x, y-1),  get(left, x+1, y),   get(right, x-1, y), labels });
                }
            }
        } else  {
            // right messages
            for (uint x = 1; x < width - 1; ++x) {
                for (uint y = 1; y < height - 1; ++y) {
                    send_msg({ ndx(u,  x, y+1),    ndx(d,    x, y-1),  ndx(r,     x-1, y),  ndx(l,    x+1, y),    ndx(pot, x, y),    ndx(r, x, y),
                                 get(up, x, y+1),    get(down, x, y-1),  get(right, x-1, y),  get(left, x+1, y), labels });
                    }
            }

            // down messages
            for (uint x = 1; x < width - 1; ++x) {
                for (uint y = 1; y < height - 1; ++y) {
                    send_msg({ ndx(d,    x, y-1),  ndx(l   , x+1, y),  ndx(r,     x-1, y),  ndx(u,  x, y+1),      ndx(pot, x, y),    ndx(d, x, y),
                             get(down, x, y-1),  get(left, x+1, y),  get(right, x-1, y),  get(up, x, y+1), labels });
                    }
            }

            // left messages
            for (uint x = 1; x < width - 1; ++x) {
                for (uint y = 1; y < height - 1; ++y) {
                    send_msg({ ndx(u,  x, y+1),    ndx(d,    x, y-1),  ndx(l,    x+1, y),   ndx(r,     x-1, y),   ndx(pot, x, y),    ndx(l, x, y),
                             get(up, x, y+1),    get(down, x, y-1),  get(left, x+1, y),   get(right, x-1, y), labels });
                }
            }

            // up messages
            for (uint x = 1; x < width - 1; ++x) {
                for (uint y = 1; y < height - 1; ++y) {
                    send_msg({ ndx(u,  x, y+1),    ndx(l   , x+1, y),  ndx(r,     x-1, y),  ndx(d,    x, y-1),    ndx(pot, x, y),    ndx(u, x, y),
                             get(up, x, y+1),    get(left, x+1, y),  get(right, x-1, y),  get(down, x, y-1), labels });
                }
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