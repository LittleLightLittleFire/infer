#include "qp.h"

#include <cassert>
#include <numeric>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace infer {

qp::qp(const crf &crf)
    : method(crf)
    , ndx_(crf_.width_, crf_.height_, crf_.labels_)
    , q_(crf_.unary_.size())
    , mu1_(crf_.unary_.size())
    , mu2_(crf_.unary_.size())
    , mu_(&mu1_[0])
    , mu_next_(&mu2_[0]) {

    // initalise with MAP
    for (unsigned y = 1; y < crf_.height_ - 1; ++y) {
        for (unsigned x = 1; x < crf_.width_ - 1; ++x) {
            float *const begin = mu_ + ndx_(x,y);
            float *const end = mu_ + ndx_(x,y) + crf_.labels_;;

            float *const min = std::min_element(begin, end);
            float *const max = std::max_element(begin, end);

            if (*max != 0) {
                for (float *i = begin; i != end; ++i) {
                    *i = (i == min) ? 1 : 0;
                }
            } else {
                *begin = 1;
            }
        }
    }

}

void qp::run(const unsigned iterations) {
    for (unsigned n = 0; n < iterations; ++n) {
        // calculate the gradient vectors
        for (unsigned y = 1; y < crf_.height_ - 1; ++y) {
            for (unsigned x = 1; x < crf_.width_ - 1; ++x) {
                for (unsigned i = 0; i < crf_.labels_; ++i) {
                    float grad = 0; // gradient of x_i

                    auto pair = [i, x, y, &grad, this](unsigned xj, unsigned yj) {
                        for (unsigned j = 0; j < crf_.labels_; ++j) {
                            grad += crf_.pairwise(x, y, i, xj, yj, j) * mu_[ndx_(xj, yj) + j];
                        }
                    };

                    // add the contributions from nearby edges
                    pair(x+1, y);
                    pair(x-1, y);
                    pair(x, y-1);
                    pair(x, y+1);

                    //std::cout << x << " " << y << " " << grad * 2 << " " << crf_.unary(x,y,i) << std::endl;
                    grad *= 2;
                    grad += crf_.unary(x, y, i);
                    q_[ndx_(x, y) + i] = grad;
                }
            }
        }

        // TODO: gradient vectors of the edges

        // calculate the new mu
        for (unsigned y = 1; y < crf_.height_ - 1; ++y) {
            for (unsigned x = 1; x < crf_.width_ - 1; ++x) {
                float pair_sum = 0;

                for (unsigned i = 0; i < crf_.labels_; ++i) {
                    const unsigned idx = ndx_(x, y) + i;
                    pair_sum += mu_[idx] * q_[idx];
                }

                for (unsigned i = 0; i < crf_.labels_; ++i) {
                    const unsigned idx = ndx_(x, y) + i;
                    if (pair_sum != 0) {
                        mu_next_[idx] = (mu_[idx] * q_[idx]) / pair_sum;
                    } else {
                        //std::cout << x << " " << y << " " << i << std::endl;
                    }
                    //std::cout << x << " " << y << " " << mu_next_[idx] << " " << pair_sum << std::endl;
                }
            }
        }

        // TODO: new_mu of the edges

        std::swap(mu_, mu_next_);
    }
}

unsigned qp::get_label(const unsigned x, const unsigned y) const {
    //unsigned min_label = 0;
    //float min_value = std::numeric_limits<float>::max();

    ////std::cout << "set: " << x << " " << y << std::endl;
    //for (unsigned i = 0; i < crf_.labels_; ++i) {
        //float val = mu_[ndx_(x, y) + i];

        //if (val < min_value) {
            //min_label = i;
            //min_value = val;
        //}
    //}
    ////std::cout << "set: " << x << " " << y << " " << std::accumulate(&mu_[ndx_(x,y)], &mu_[ndx_(x,y) + crf_.labels_], 0.0f) << std::endl;

    //return min_label;

    unsigned max_label = 0;
    float max_value = std::numeric_limits<float>::lowest();

    for (unsigned i = 0; i < crf_.labels_; ++i) {
        float val = mu_[ndx_(x, y) + i];

        if (val > max_value) {
            max_label = i;
            max_value = val;
        }
    }
    //std::cout << "set: " << x << " " << y << " " << std::accumulate(&mu_[ndx_(x,y)], &mu_[ndx_(x,y) + crf_.labels_], 0.0f) << std::endl;
    return max_label;
}

}
