#include "cuda/bp.h"
#include "cuda/util.h"

#include "cuda/core.h"

namespace infer {
namespace cuda {

bp::bp(const crf &crf)
    : method(crf)
    , current_iteration(0)
    , dev_l_(0), dev_r_(0), dev_u_(0), dev_d_(0) {

    const size_t size = crf.width_ * crf.height_ * crf.labels_ * sizeof(float);
    cuda_check(cudaMalloc(&dev_l_, size));
    cuda_check(cudaMalloc(&dev_r_, size));
    cuda_check(cudaMalloc(&dev_u_, size));
    cuda_check(cudaMalloc(&dev_d_, size));

    cuda_check(cudaMemset(dev_l_, 0, size));
    cuda_check(cudaMemset(dev_r_, 0, size));
    cuda_check(cudaMemset(dev_u_, 0, size));
    cuda_check(cudaMemset(dev_d_, 0, size));
}

void bp::run(const unsigned iterations) {
    if (iterations != 0) {
        dirty_ = true;
    }

    dim3 block(8, 16); // only half of the pixels are updated because of the checkboard pattern
    dim3 grid(((crf_.width_ + 1) / 2 + block.x - 1) / block.x, (crf_.height_ + block.y - 1) / block.y);

    for (unsigned i = 0; i < iterations; ++i) {
        ++current_iteration;

        trbp_run<<<grid, block>>>(crf_.labels_, crf_.width_, crf_.height_, current_iteration, crf_.type_, crf_.lambda_, crf_.trunc_, crf_.dev_pairwise_, dev_l_, dev_r_, dev_u_, dev_d_, crf_.dev_unary_, 0);
        cuda_check(cudaGetLastError());
    }
}

void bp::update_dev_result() const {
    dim3 block(16, 16);
    dim3 grid((crf_.width_ + block.x - 1) / block.x, (crf_.height_ + block.y - 1) / block.y);
    trbp_get_results<<<grid, block>>>(crf_.labels_, crf_.width_, crf_.height_, dev_l_, dev_r_, dev_u_, dev_d_, crf_.dev_unary_, dev_result_, 0);
    cuda_check(cudaGetLastError());
}

std::string bp::get_name() const {
    return "gpu_bp";
}

bp::~bp() {
    cudaFree(dev_l_);
    cudaFree(dev_r_);
    cudaFree(dev_u_);
    cudaFree(dev_d_);
}

}
}
