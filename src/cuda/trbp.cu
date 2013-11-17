#include "cuda/trbp.h"
#include "cuda/util.h"

#include "cuda/core.h"

namespace infer {
namespace cuda {

trbp::trbp(const crf &crf, const std::vector<float> rho)
    : bp(crf)
    , dev_rho_(0) {

    cuda_check(cudaMalloc(&dev_rho_, rho.size() * sizeof(float)));
    cuda_check(cudaMemcpy(dev_rho_, &rho[0], rho.size() * sizeof(float)));
}

void trbp::run(const unsigned iterations) {
}

void trbp::update_dev_result() const {
}

trbp::~trbp() {
    cudaFree(dev_rho_);
}

}
}
