#include "method.h"
#include "util.h"

#include <iostream>

namespace infer {

method::method(const crf &crf)
    : crf_(crf) {
}

std::vector<unsigned> method::get_result() const {
    std::vector<unsigned> result;
    for (unsigned y = 0; y < crf_.height_; ++y) {
        for (unsigned x = 0; x < crf_.width_; ++x) {
            result.push_back(get_label(x, y));
        }
    }

    return result;
}

/** TODO: remove once gcc is updated, clang needs this method to link properly */
method::~method() {
}

}
