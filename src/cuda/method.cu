#include "cuda/method.h"
#include "cuda/util.h"

namespace infer {
namespace cuda {

method::method(const crf &crf)
    : crf_(crf)
    , dev_result_(0)
    , dirty_(true) {
    cuda_check(cudaMalloc(&dev_result_, crf.width_ * crf.height_ * sizeof(unsigned)));
}

std::vector<unsigned> method::get_result() const {
    if (dirty_) {
        update_dev_result();
        dirty_ = false;
    }

    std::vector<unsigned> result(crf_.width_ * crf_.height_);
    cuda_check(cudaMemcpy(&result[0], dev_result_, result.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
    return result;
}

method::~method() {
    cudaFree(dev_result_);
}

}
}
