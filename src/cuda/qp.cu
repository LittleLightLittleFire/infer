#include "cuda/qp.h"
#include "cuda/util.h"

#include "cuda/core.h"

#include <cuda.h>

namespace infer {
namespace cuda {

qp::qp(const crf &crf)
    : method(crf)
    , dev_mu1_(0)
    , dev_mu2_(0) {

    const size_t size = crf.width_ * crf.height_ * crf.labels_ * sizeof(float);
    cuda_check(cudaMalloc(&dev_mu1_, size));
    cuda_check(cudaMalloc(&dev_mu2_, size));

    // initalise with -ve exp
    dim3 block(16, 16);
    dim3 grid((crf_.width_ + block.x - 1) / block.x, (crf_.height_ + block.y - 1) / block.y);
    qp_initalise<<<grid, block>>>(crf_.labels_, crf_.width_, crf_.height_, crf.dev_unary_, dev_mu1_);
    cuda_check(cudaGetLastError());
}

void qp::update_dev_result() const {
    dim3 block(16, 16);
    dim3 grid((crf_.width_ + block.x - 1) / block.x, (crf_.height_ + block.y - 1) / block.y);

    qp_get_results<<<grid, block>>>(crf_.labels_, crf_.width_, crf_.height_, dev_mu1_, dev_result_);
    cuda_check(cudaGetLastError());
}

void qp::run(const unsigned iterations) {
    if (iterations != 0) {
        dirty_ = true;
    }

    dim3 block(16, 16);
    dim3 grid((crf_.width_ + block.x - 1) / block.x, (crf_.height_ + block.y - 1) / block.y);

    for (unsigned i = 0; i < iterations; ++i) {
        qp_run<<<grid, block>>>(crf_.labels_, crf_.width_, crf_.height_, crf_.dev_unary_, crf_.lambda_, crf_.dev_pairwise_, crf_.type_, dev_mu1_, dev_mu2_);
        cuda_check(cudaGetLastError());

        std::swap(dev_mu1_, dev_mu2_);
    }
}

std::string qp::get_name() const {
    return "gpu_qp";
}

qp::~qp() {
    cudaFree(dev_mu1_);
    cudaFree(dev_mu2_);
}

}
}
