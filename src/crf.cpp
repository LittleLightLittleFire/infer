#include "crf.h"

#include <cmath>

namespace infer {

crf::crf(const unsigned width, const unsigned height, const unsigned labels, const std::vector<float> unary, const float lambda, const std::vector<float> pairwise)
    : width_(width)
    , height_(height)
    , labels_(labels)
    , type_(type::ARRAY)
    , unary_(unary)
    , lambda_(lambda)
    , trunc_(0)
    , pairwise_(pairwise)
    , idx_(width_, height_), ndx_(width_, height_, labels_), edx_(width_, height_) {
}

crf::crf(const unsigned width, const unsigned height, const unsigned labels, const std::vector<float> unary, const float lambda, const unsigned norm, const unsigned trunc)
    : width_(width)
    , height_(height)
    , labels_(labels)
    , type_(norm == 1 ? type::L1 : type::L2)
    , unary_(unary)
    , lambda_(lambda)
    , trunc_(trunc)
    , pairwise_()
    , idx_(width_, height_), ndx_(width_, height_, labels_), edx_(width_, height_) {
}

const float *crf::unary(const unsigned x, const unsigned y) const {
    return &unary_[ndx_(x,y)];
}

float crf::unary(const unsigned x, const unsigned y, const unsigned label) const {
    return unary_[ndx_(x,y) + label];
}

float crf::pairwise(const unsigned x, const unsigned y, const unsigned l1, const move dir, const unsigned l2) const {
    switch (type_) {
        case type::ARRAY:
            return lambda_ * pairwise_[edx_(x, y, dir) * l2 * labels_ + l1];
        case type::L1:
            return lambda_ * std::min(std::abs(static_cast<float>(l1) - static_cast<float>(l2)), trunc_);
        case type::L2:
            const float tmp = std::abs(static_cast<float>(l1) - static_cast<float>(l2));
            return lambda_ * std::min(tmp * tmp, trunc_);
    }
}

float crf::unary_energy(const std::vector<unsigned> labeling) const {
    float unary_energy = 0;

    // node potentials
    for (unsigned y = 0; y < height_; ++y) {
        for (unsigned x = 0; x < width_; ++x) {
            unary_energy += unary(x, y, labeling[idx_(x, y)]);
        }
    }

    return unary_energy;
}

float crf::pairwise_energy(const std::vector<unsigned> labeling) const {
    float pairwise_energy = 0;
    // edge energy
    for (unsigned y = 0; y < height_ - 1; ++y) {
        for (unsigned x = 0; x < width_ - 1; ++x) {
            pairwise_energy += pairwise(x, y, labeling[idx_(x, y)], move::RIGHT, labeling[idx_(x+1, y)]);
            pairwise_energy += pairwise(x, y, labeling[idx_(x, y)], move::DOWN, labeling[idx_(x, y+1)]);
        }
    }

    return pairwise_energy;
}

crf crf::downsize() const {
    const unsigned new_width = width_ / 2, new_height = height_ / 2;

    // sum of potentials in a 2x2 square
    std::vector<float> new_unary(new_width * new_height * labels_);
    const indexer ndx(new_width, new_height, labels_);

    for (unsigned y = 0; y < height_; ++y) {
        for (unsigned x = 0; x < width_; ++x) {
            for (unsigned j = 0; j < labels_; ++j) {
                new_unary[ndx(x / 2, y / 2) + j] += unary(x, y, j);
            }
        }
    }

    if (type_ == crf::type::ARRAY) {
        return crf(new_width, new_height, labels_, new_unary, lambda_, pairwise_);
    } else {
        const unsigned norm = type_ == crf::type::L1 ? 1 : 2;
        return crf(new_width, new_height, labels_, new_unary, lambda_, norm, trunc_);
    }
}

}
