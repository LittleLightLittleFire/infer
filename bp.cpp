#include "bp.h"

#include <limits>
#include <algorithm>

namespace infer {

namespace {

inline void send_msg(const crf &crf_, const float *m1, const float *m2, const float *m3, const float *pot, float *out, const unsigned x, const unsigned y, const unsigned xt, const unsigned yt) {
    const unsigned labels = crf_.labels_;
    switch (crf_.type_) {
        case crf::type::L1:
            // use the O(n) algorithm from Pedro F. Felzenszwalb and Daniel P. Huttenlocher (2006): Efficient Belief Propagation for Early Vision
            // compute the new message partially, deal with the pairwise term later
            for (unsigned i = 0; i < labels; ++i) {
                out[i] = m1[i] + m2[i] + m3[i] + pot[i];
            }

            { // truncate
                // compute the minimum to truncate with
                const float trunc = crf_.trunc_ * crf_.lambda_;
                const float scale = crf_.lambda_;

                const float trunc_min = trunc + *std::min_element(out, out + labels);

                for (unsigned i = 1; i < labels; ++i) {
                    out[i] = std::min(out[i - 1] + scale, out[i]);
                }

                // second pass, same th but with the list reversed
                for (unsigned i = labels - 2; i-- > 0; ) {
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
                    const float val = pot[j] + m1[j] + m2[j] + m3[j] + crf_.pairwise(x, y, i, xt, yt, j);
                    out[i] = std::min(out[i], val);
                }
            }
            break;
    }

    // normalise floating point messages to avoid over/underflow
    const float sum = out[0];//std::accumulate(out, out + labels, 0.0f) / static_cast<float>(labels);
    std::transform(out, out + labels, out, [sum](const float x){ return x - sum; });
}

}

bp::bp(const crf &crf, const bool synchronous)
    : method(crf)
    , synchronous_(synchronous)
    , current_iter(0)
    , ndx_(crf_.width_, crf_.height_, crf.labels_)
    , up_(crf_.width_ * crf_.height_ * crf.labels_)
    , down_(crf_.width_ * crf_.height_ * crf.labels_)
    , left_(crf_.width_ * crf_.height_ * crf.labels_)
    , right_(crf_.width_ * crf_.height_ * crf.labels_) {
}

float bp::msg(const std::vector<float> &msg, const unsigned x, const unsigned y, const unsigned label) const {
    return msg[ndx_(x, y) + label];
}

float *bp::msg(std::vector<float> &msg, const unsigned x, const unsigned y) const {
    return &msg[ndx_(x, y)];
}

void bp::run(const unsigned iterations) {
    for (unsigned i = 0; i < iterations; ++i) {
        ++current_iter;

        if (synchronous_) {
            // do not use i for when doing the checkboard update pattern
            for (unsigned y = 1; y < crf_.height_ - 1; ++y) {
                for (unsigned x = ((y + current_iter) % 2) + 1; x < crf_.width_ - 1; x += 2) {
                    // send messages in each direction
                    //        m1                       m2                  m3                   pot               out
                    send_msg(crf_, msg(up_,   x, y+1), msg(down_, x, y-1), msg(right_, x-1, y), crf_.unary(x, y), msg(right_, x, y), x, y, x+1, y);
                    send_msg(crf_, msg(up_,   x, y+1), msg(down_, x, y-1), msg(left_,  x+1, y), crf_.unary(x, y), msg(left_, x, y),  x, y, x-1, y);
                    send_msg(crf_, msg(down_, x, y-1), msg(left_, x+1, y), msg(right_, x-1, y), crf_.unary(x, y), msg(down_, x, y),  x, y, x, y+1);
                    send_msg(crf_, msg(up_,   x, y+1), msg(left_, x+1, y), msg(right_, x-1, y), crf_.unary(x, y), msg(up_, x, y),    x, y, x, y-1);
                }
            }
        } else {
            const unsigned width = crf_.width_;
            const unsigned height = crf_.height_;

            // right and left messages
            for (unsigned y = 1; y < height - 1; ++y) {
                for (unsigned x = 1; x < width - 1; ++x) {
                    send_msg(crf_, msg(up_,   x, y+1), msg(down_, x, y-1), msg(right_, x-1, y), crf_.unary(x, y), msg(right_, x, y), x, y, x+1, y);
                }
                for (unsigned x = width - 1; x-- > 1; ) {
                    send_msg(crf_, msg(up_,   x, y+1), msg(down_, x, y-1), msg(left_,  x+1, y), crf_.unary(x, y), msg(left_, x, y),  x, y, x-1, y);
                }
            }

            // down and up messages
            for (unsigned x = 1; x < width - 1; ++x) {
                for (unsigned y = 1; y < height - 1; ++y) {
                    send_msg(crf_, msg(down_, x, y-1), msg(left_, x+1, y), msg(right_, x-1, y), crf_.unary(x, y), msg(down_, x, y),  x, y, x, y+1);
                }
                for (unsigned y = height - 1; y-- > 1; ) {
                    send_msg(crf_, msg(up_,   x, y+1), msg(left_, x+1, y), msg(right_, x-1, y), crf_.unary(x, y), msg(up_, x, y),    x, y, x, y-1);
                }
            }
        }
    }
}

unsigned bp::get_label(const unsigned x, const unsigned y) const {
    // the label of the node is the label with the lowest energy
    unsigned min_label = 0;
    float min_value = std::numeric_limits<float>::max();



    for (unsigned i = 0; i < crf_.labels_; ++i) {
        float val = crf_.unary(x, y, i);

        // check the bounds first
        if (y + 1 < crf_.height_) val += msg(up_, x, y+1, i);
        if (x + 1 < crf_.width_)  val += msg(left_, x+1, y, i);
        if (y != 0)               val += msg(down_, x, y-1, i);
        if (x != 0)               val += msg(right_, x-1, y, i);

        if (val < min_value) {
            min_label = i;
            min_value = val;
        }
    }
    return min_label;
}

}
