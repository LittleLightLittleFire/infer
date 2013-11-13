#include "method.h"
#include "util.h"

namespace infer {

method::method(const crf &crf)
    : crf_(crf) {
}


std::tuple<float, float> method::get_energy() const {
    float unary_energy = 0;
    float pairwise_energy = 0;

    // node potentials
    for (unsigned y = 0; y < crf_.height_; ++y) {
        for (unsigned x = 0; x < crf_.width_; ++x) {
            unary_energy += crf_.unary(x, y, get_label(x, y));
        }
    }

    // edge energy
    for (unsigned y = 0; y < crf_.height_ - 1; ++y) {
        for (unsigned x = 0; x < crf_.width_ - 1; ++x) {
            pairwise_energy += crf_.pairwise(x, y, get_label(x, y), x+1, y, get_label(x+1, y));
            pairwise_energy += crf_.pairwise(x, y, get_label(x, y), x, y+1, get_label(x, y+1));
        }
    }

    // energy of the right edge
    for (unsigned y = 0; y < crf_.height_ - 1; ++y) {
        const unsigned x = crf_.width_ - 1;
        pairwise_energy += crf_.pairwise(x, y, get_label(x, y), x, y+1, get_label(x, y+1));
    }

    // energy of bottom edge
    for (unsigned x = 0; x < crf_.width_ - 1; ++x) {
        const unsigned y = crf_.height_ - 1;
        pairwise_energy += crf_.pairwise(x, y, get_label(x, y), x+1, y, get_label(x+1, y));
    }

    return std::make_tuple(unary_energy, pairwise_energy);
}

}
