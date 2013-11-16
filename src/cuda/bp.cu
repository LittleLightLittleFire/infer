#include "cuda/bp.h"

namespace infer {
namespace cuda {

bp::bp(const crf &crf)
    : method(crf) {
}

void bp::run(const unsigned iterations) {
}

float bp::unary_energy() const {
    return 1;
}

float bp::pairwise_energy() const {
    return 2;
}

std::vector<unsigned> bp::get_result() const {
    return std::vector<unsigned>(crf_.width_ * crf_.height_);
}

bp::~bp() {
}

}
}
