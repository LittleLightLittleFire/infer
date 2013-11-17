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
                    const float val = pot[j] + m1[j] + m2[j] + m3[j] + crf_.pairwise(x, y, i, xt, yt, j);
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

bp::bp(const crf &crf, const bp &prev)
    : bp(crf, prev.synchronous_) {

    const indexer pdx(prev.crf_.width_, prev.crf_.height_, prev.crf_.labels_);

    // set the messages using prev
    for (unsigned y = 0; y < crf.height_; ++y) {
        for (unsigned x = 0; x < crf.width_; ++x) {
            for (unsigned i = 0; i < crf.labels_; ++i) {
                const unsigned nidx = ndx_(x, y) + i;
                const unsigned pidx = pdx(x / 2, y / 2) + i;

                up_[nidx] = prev.up_[pidx];
                down_[nidx] = prev.down_[pidx];
                left_[nidx] = prev.left_[pidx];
                right_[nidx] = prev.right_[pidx];
            }
        }
    }
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
                    // send messages out of x,y
                    //        m1                       m2                  m3                   pot               out
                    send_msg(crf_, msg(up_,   x, y), msg(down_, x, y), msg(right_, x, y), crf_.unary(x, y), msg(right_, x+1, y), x, y, x+1, y);
                    send_msg(crf_, msg(up_,   x, y), msg(down_, x, y), msg(left_,  x, y), crf_.unary(x, y), msg(left_,  x-1, y), x, y, x-1, y);
                    send_msg(crf_, msg(down_, x, y), msg(left_, x, y), msg(right_, x, y), crf_.unary(x, y), msg(down_,  x, y+1), x, y, x, y+1);
                    send_msg(crf_, msg(up_,   x, y), msg(left_, x, y), msg(right_, x, y), crf_.unary(x, y), msg(up_,    x, y-1), x, y, x, y-1);
                }
            }
        } else {
            const unsigned width = crf_.width_;
            const unsigned height = crf_.height_;

            // right and left messages
            for (unsigned y = 0; y < height; ++y) {
                for (unsigned x = 0; x < width - 1; ++x) {
                    send_msg(crf_, msg(up_,   x, y), msg(down_, x, y), msg(right_, x, y), crf_.unary(x, y), msg(right_, x+1, y), x, y, x+1, y);
                }
                for (unsigned x = width - 1; x > 0; --x) {
                    send_msg(crf_, msg(up_,   x, y), msg(down_, x, y), msg(left_,  x, y), crf_.unary(x, y), msg(left_,  x-1, y), x, y, x-1, y);
                }
            }

            // down and up messages
            for (unsigned x = 0; x < width; ++x) {
                for (unsigned y = 0; y < height - 1; ++y) {
                    send_msg(crf_, msg(down_, x, y), msg(left_, x, y), msg(right_, x, y), crf_.unary(x, y), msg(down_,  x, y+1), x, y, x, y+1);
                }
                for (unsigned y = height - 1; y > 0; --y) {
                    send_msg(crf_, msg(up_,   x, y), msg(left_, x, y), msg(right_, x, y), crf_.unary(x, y), msg(up_,    x, y-1), x, y, x, y-1);
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
        const float val = crf_.unary(x, y, i)
                        + msg(up_, x, y, i)
                        + msg(left_, x, y, i)
                        + msg(down_, x, y, i)
                        + msg(right_, x, y, i);

        if (val < min_value) {
            min_label = i;
            min_value = val;
        }
    }
    return min_label;
}

}
