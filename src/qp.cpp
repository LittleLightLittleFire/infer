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
    , mu1_(crf_.unary_)
    , mu2_(crf_.unary_.size())
    , mu_(&mu1_[0])
    , mu_next_(&mu2_[0]) {

    for (unsigned y = 1; y < crf_.height_ - 1; ++y) {
        for (unsigned x = 1; x < crf_.width_ - 1; ++x) {
            float *const begin = mu_ + ndx_(x,y);
            float *const end = begin + crf_.labels_;;

            float total = 0;
            for (float *i = begin; i != end; ++i) {
                *i = std::exp(-*i);
                total += *i;
            }

            if (total == 0) {
                for (float *i = begin; i != end; ++i) {
                    *i = 1 / static_cast<float>(crf_.labels_);
                }
            } else {
                for (float *i = begin; i != end; ++i) {
                    *i /= total;
                }
            }
        }
    }

}

void qp::run(const unsigned iterations) {
    for (unsigned n = 0; n < iterations; ++n) {
        // calculate the gradient vectors
        for (unsigned y = 1; y < crf_.height_ - 1; ++y) {
            for (unsigned x = 1; x < crf_.width_ - 1; ++x) {
                float total = 0;

                for (unsigned i = 0; i < crf_.labels_; ++i) {
                    float grad = 0; // gradient of x_i

                    auto pair = [i, x, y, &grad, this](const unsigned xj, const unsigned yj, const move m) {
                        for (unsigned j = 0; j < crf_.labels_; ++j) {
                            grad += std::exp(- crf_.pairwise(x, y, i, m, j)) * mu_[ndx_(xj, yj, j)];
                        }
                    };

                    // add the contributions from nearby edges
                    pair(x+1, y, move::RIGHT);
                    pair(x-1, y, move::LEFT);
                    pair(x, y-1, move::UP);
                    pair(x, y+1, move::DOWN);

                    grad *= 2;
                    grad += std::exp(- crf_.unary(x, y, i));

                    assert(mu_[ndx_(x, y, i)] >= 0); assert(mu_[ndx_(x, y, i)] <= 1);
                    total += mu_next_[ndx_(x, y, i)] = mu_[ndx_(x, y, i)] * grad;
                }

                for (unsigned i = 0; i < crf_.labels_; ++i) {
                    mu_next_[ndx_(x, y, i)] /= total;
                }
            }
        }

        std::swap(mu_, mu_next_);
    }
}

float qp::objective() const { float obj = 0;
    float obj_pair = 0;
    for (unsigned y = 1; y < crf_.height_ - 1; ++y) {
        for (unsigned x = 1; x < crf_.width_ - 1; ++x) {
            for (unsigned i = 0; i < crf_.labels_; ++i) {
                assert(mu_[ndx_(x, y, i)] >= 0); assert(mu_[ndx_(x, y, i)] <= 1);

                obj += mu_[ndx_(x, y, i)] * crf_.unary(x, y, i);

                for (unsigned j = 0; j < crf_.labels_; ++j) {
                    obj_pair += mu_[ndx_(x, y, i)] * mu_[ndx_(x, y + 1, j)] * crf_.pairwise(x, y, i, move::DOWN, j);
                    obj_pair += mu_[ndx_(x, y, i)] * mu_[ndx_(x + 1, y, j)] * crf_.pairwise(x, y, i, move::RIGHT, j);
                }
            }
        }
    }

    //std::cout << obj << " " << obj_pair << " ";

    return obj + obj_pair;
}

unsigned qp::get_label(const unsigned x, const unsigned y) const {
    //std::cout << "set: " << x << " " << y << " " << std::accumulate(&mu_[ndx_(x,y)], &mu_[ndx_(x,y) + crf_.labels_], 0.0f) << std::endl;

    unsigned max_label = 0;
    float max_value = std::numeric_limits<float>::lowest();

    for (unsigned i = 0; i < crf_.labels_; ++i) {
        float val = mu_[ndx_(x, y) + i];
        //std::cout << x << " " << y << " " << i << " " << val << std::endl;

        if (val > max_value) {
            max_label = i;
            max_value = val;
        }
    }
    //std::cout << "set: " << x << " " << y << " " << std::accumulate(&mu_[ndx_(x,y)], &mu_[ndx_(x,y) + crf_.labels_], 0.0f) << std::endl;
    return max_label;
}

std::string qp::get_name() const {
    return "qp";
}

}
