#include "trbp.h"

#include <limits>
#include <algorithm>

namespace infer {

namespace {

inline void send_msg(const crf &crf_
                   , const float *m1, const float *m2, const float *m3, const float *opp
                   , const float rm1, const float rm2, const float rm3, const float ropp
                   , const float *pot, float *out
                   , const unsigned x, const unsigned y, const move m) {

    const unsigned labels = crf_.labels_;

    switch (crf_.type_) {
        case crf::type::L1:
            // use the O(n) algorithm from Pedro F. Felzenszwalb and Daniel P. Huttenlocher (2006): Efficient Belief Propagation for Early Vision
            // compute the new message partially, deal with the pairwise term later
            for (unsigned i = 0; i < labels; ++i) {
                out[i] = pot[i] + m1[i] * rm1 + m2[i] * rm2 + m3[i] * rm3 - opp[i] * (1 - ropp);
            }

            { // truncate
                // compute the minimum to truncate with
                const float trunc = crf_.trunc_ * crf_.lambda_ * (1 / ropp);
                const float scale = crf_.lambda_ * (1 / ropp);

                const float trunc_min = trunc + *std::min_element(out, out + labels);

                for (unsigned i = 1; i < labels; ++i) {
                    out[i] = std::min(out[i - 1] + scale, out[i]);
                }

                for (unsigned i = labels - 1; i-- > 0; ) {
                    out[i] = std::min(out[i + 1] + scale, out[i]);
                }

                std::transform(out, out + labels, out, [trunc_min](const float x){ return std::min(x, trunc_min); });
            }
            break;
        case crf::type::L2: // TODO: optimised L2 norm algorithm
        case crf::type::ARRAY:
        default:
            for (unsigned i = 0; i < labels; ++i) {
                out[i] = std::numeric_limits<float>::max();
                for (unsigned j = 0; j < labels; ++j) {
                    const float val = pot[j] + m1[j] * rm1 + m2[j] * rm2 + m3[j] * rm3 - opp[j] * (1 - ropp) + (1 / ropp) * crf_.pairwise(x, y, i, m, j);
                    out[i] = std::min(out[i], val);
                }
            }
            break;
    }

    // normalise floating point messages to avoid over/underflow
    const float first = out[0];
    std::transform(out, out + labels, out, [first](const float x){ return x - first; });
}
}

trbp::trbp(const crf &crf, const std::vector<float> rho, const bool synchronous)
    : bp(crf, synchronous)
    , rho_(rho)
    , edx_(crf_.width_, crf_.height_) {

}

void trbp::run(const unsigned iterations) {
    // convenience method to get the edge probabilitiy
    // since msg are the messages coming into the node, the directions are opposite
    const move left = move::RIGHT, right = move::LEFT, up = move::DOWN, down = move::UP;
    auto get = [this](const move m, const unsigned x, const unsigned y) {
        return rho_[edx_(x, y, m)];
    };

    for (unsigned i = 0; i < iterations; ++i) {
        ++current_iter;

        if (synchronous_) {
            // do not use i for when doing the checkboard update pattern
            for (unsigned y = 1; y < crf_.height_ - 1; ++y) {
                for (unsigned x = ((y + current_iter) % 2) + 1; x < crf_.width_ - 1; x += 2) {
                    // send messages out of x,y
                    //                   m1                 m2                  m3              opp
                    send_msg(crf_, msg(up_,   x, y), msg(down_, x, y), msg(right_, x, y), msg(left_,  x, y)
                                 , get(up,    x, y), get(down,  x, y), get(right,  x, y), get(left,   x, y), crf_.unary(x, y), msg(right_, x+1, y), x, y, move::RIGHT);

                    send_msg(crf_, msg(up_,   x, y), msg(down_, x, y), msg(left_,  x, y), msg(right_, x, y)
                                 , get(up,    x, y), get(down,  x, y), get(left,   x, y), get(right,  x, y), crf_.unary(x, y), msg(left_,  x-1, y), x, y, move::LEFT);

                    send_msg(crf_, msg(down_, x, y), msg(left_, x, y), msg(right_, x, y), msg(up_,    x, y)
                                 , get(down,  x, y), get(left,  x, y), get(right,  x, y), get(up,     x, y), crf_.unary(x, y), msg(down_,  x, y+1), x, y, move::DOWN);

                    send_msg(crf_, msg(up_,   x, y), msg(left_, x, y), msg(right_, x, y), msg(down_,  x, y)
                                 , get(up,    x, y), get(left,  x, y), get(right,  x, y), get(down,   x, y), crf_.unary(x, y), msg(up_,    x, y-1), x, y, move::UP);
                }
            }
        } else {
            const unsigned width = crf_.width_;
            const unsigned height = crf_.height_;

            // right and left messages
            for (unsigned y = 1; y < height - 1; ++y) {
                for (unsigned x = 1; x < width - 1; ++x) {
                    send_msg(crf_, msg(up_,   x, y), msg(down_, x, y), msg(right_, x, y), msg(left_,  x, y)
                                 , get(up,    x, y), get(down,  x, y), get(right,  x, y), get(left,   x, y), crf_.unary(x, y), msg(right_, x+1, y), x, y, move::RIGHT);
                }
                for (unsigned x = width - 2; x > 0; --x) {
                    send_msg(crf_, msg(up_,   x, y), msg(down_, x, y), msg(left_,  x, y), msg(right_, x, y)
                                 , get(up,    x, y), get(down,  x, y), get(left,   x, y), get(right,  x, y), crf_.unary(x, y), msg(left_,  x-1, y), x, y, move::LEFT);
                }
            }

            // down and up messages
            for (unsigned x = 1; x < width - 1; ++x) {
                for (unsigned y = 1; y < height - 1; ++y) {
                    send_msg(crf_, msg(down_, x, y), msg(left_, x, y), msg(right_, x, y), msg(up_,    x, y)
                                 , get(down,  x, y), get(left,  x, y), get(right,  x, y), get(up,     x, y), crf_.unary(x, y), msg(down_,  x, y+1), x, y, move::DOWN);
                }
                for (unsigned y = height - 2; y > 0; --y) {
                    send_msg(crf_, msg(up_,   x, y), msg(left_, x, y), msg(right_, x, y), msg(down_,  x, y)
                                 , get(up,    x, y), get(left,  x, y), get(right,  x, y), get(down,   x, y), crf_.unary(x, y), msg(up_,    x, y-1), x, y, move::UP);
                }
            }
        }
    }
}

unsigned trbp::get_label(const unsigned x, const unsigned y) const {
    // the label of the node is the label with the lowest energy
    unsigned min_label = 0;
    float min_value = std::numeric_limits<float>::max();

    for (unsigned i = 0; i < crf_.labels_; ++i) {
        float val = crf_.unary(x, y, i);

        if (y != crf_.height_ - 1) val += msg(up_, x, y, i)    * rho_[edx_(x, y, move::DOWN)];
        if (x != crf_.width_ - 1)  val += msg(left_, x, y, i)  * rho_[edx_(x, y, move::RIGHT)];
        if (y != 0)                val += msg(down_, x, y, i)  * rho_[edx_(x, y, move::UP)];
        if (x != 0)                val += msg(right_, x, y, i) * rho_[edx_(x, y, move::LEFT)];

        if (val < min_value) {
            min_label = i;
            min_value = val;
        }
    }
    return min_label;
}

std::string trbp::get_name() const {
    return synchronous_ ? "trbp" : "trbp_async";
}

}
