#include "cuda/trbp.h"
#include "cuda/util.h"

#include "cuda/core.h"

namespace infer {
namespace cuda {

trbp::trbp(const crf &crf, const std::vector<float> rho)
    : bp(crf)
    , dev_rho_(0) {

    cuda_check(cudaMalloc(&dev_rho_, rho.size() * sizeof(float)));
    cuda_check(cudaMemcpy(dev_rho_, &rho[0], rho.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void trbp::run(const unsigned iterations) {
    if (iterations != 0) {
        dirty_ = true;
    }

    dim3 block(8, 16); // only half of the pixels are updated because of the checkboard pattern
    dim3 grid(((crf_.width_ + 1) / 2 + block.x - 1) / block.x, (crf_.height_ + block.y - 1) / block.y);

    for (unsigned i = 0; i < iterations; ++i) {
        ++current_iteration_;

        trbp_run<<<grid, block>>>(crf_.labels_, crf_.width_, crf_.height_, current_iteration_, crf_.type_, crf_.lambda_, crf_.trunc_, crf_.dev_pairwise_, dev_l_, dev_r_, dev_u_, dev_d_, crf_.dev_unary_, dev_rho_);
        cuda_check(cudaGetLastError());
    }
}

void trbp::update_dev_result() const {
    dim3 block(16, 16);
    dim3 grid((crf_.width_ + block.x - 1) / block.x, (crf_.height_ + block.y - 1) / block.y);
    trbp_get_results<<<grid, block>>>(crf_.labels_, crf_.width_, crf_.height_, dev_l_, dev_r_, dev_u_, dev_d_, crf_.dev_unary_, dev_result_, dev_rho_);
    cuda_check(cudaGetLastError());
}

std::string trbp::get_name() const {
    return "gpu_trbp";
}

trbp::~trbp() {
    cudaFree(dev_rho_);
}

}
}
