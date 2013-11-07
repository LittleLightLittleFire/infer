#include "crf.h"

#include <cmath>

namespace infer {

crf::crf(const unsigned width, const unsigned height, const unsigned labels, const std::vector<float> unary, const float lambda, const std::vector<float> pairwise)
    : width_(width)
    , height_(height)
    , labels_(labels)
    , type_(type::ARRAY)
    , lambda_(lambda)
    , trunc_(0)
    , unary_(unary)
    , pairwise_(pairwise)
    , idx_(width_, height_), ndx_(width_, height_, labels_) {
}

crf::crf(const unsigned width, const unsigned height, const unsigned labels, const std::vector<float> unary, const float lambda, const unsigned norm, const unsigned trunc)
    : width_(width)
    , height_(height)
    , labels_(labels)
    , type_(norm == 1 ? type::L1 : type::L2)
    , lambda_(lambda)
    , trunc_(trunc)
    , unary_(unary)
    , pairwise_()
    , idx_(width_, height_), ndx_(width_, height_, labels_) {
}

const float *crf::unary(const unsigned x, const unsigned y) const {
    return &unary_[ndx_(x,y)];
}

float crf::unary(const unsigned x, const unsigned y, const unsigned label) const {
    return unary_[ndx_(x,y) + label];
}

float crf::pairwise(const unsigned x1, const unsigned y1, const float l1, const unsigned x2, const unsigned y2, const float l2) const {
    switch (type_) {
        case type::ARRAY:
            return lambda_ * pairwise_[l2 * labels_ + l1];
        case type::L1:
            return lambda_ * std::abs(static_cast<float>(l1) - static_cast<float>(l2));
        case type::L2:
            const float tmp = std::abs(static_cast<float>(l1) - static_cast<float>(l2));
            return lambda_ * tmp * tmp;
    }
}

}
