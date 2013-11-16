#include "method.h"
#include "util.h"

#include <iostream>

namespace infer {

method::method(const crf &crf)
    : crf_(crf) {
}

float method::unary_energy() const {
    float unary_energy = 0;

    // node potentials
    for (unsigned y = 0; y < crf_.height_; ++y) {
        for (unsigned x = 0; x < crf_.width_; ++x) {
            unary_energy += crf_.unary(x, y, get_label(x, y));
        }
    }

    return unary_energy;
}

float method::pairwise_energy() const {
    float pairwise_energy = 0;
    // edge energy
    for (unsigned y = 0; y < crf_.height_ - 1; ++y) {
        for (unsigned x = 0; x < crf_.width_ - 1; ++x) {
            pairwise_energy += crf_.pairwise(x, y, get_label(x, y), x+1, y, get_label(x+1, y));
            pairwise_energy += crf_.pairwise(x, y, get_label(x, y), x, y+1, get_label(x, y+1));
        }
    }

    return pairwise_energy;
}

std::vector<unsigned> method::get_label() const {
    std::vector<unsigned> result;
    for (unsigned y = 0; y < crf_.height_; ++y) {
        for (unsigned x = 0; x < crf_.width_; ++x) {
            result.push_back(get_label(x, y));
        }
    }

    return result;
}

}
