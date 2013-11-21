#include "cuda/crf.h"
#include "cuda/util.h"
#include "cuda/core.h"

#include <cuda.h>
#include <stdexcept>

namespace infer {
namespace cuda {

crf::crf(const unsigned width, const unsigned height, const unsigned labels, const std::vector<float> unary, const float lambda, const unsigned norm, const unsigned trunc)
    : width_(width)
    , height_(height)
    , labels_(labels)
    , dev_unary_(0)
    , lambda_(lambda)
    , type_(norm == 1 ? L1 : L2)
    , trunc_(trunc)
    , dev_pairwise_(0) {

    cuda_check(cudaMalloc(&dev_unary_, width * height * labels * sizeof(float)));
    cuda_check(cudaMemcpy(dev_unary_, &unary[0], width * height * labels * sizeof(float), cudaMemcpyHostToDevice));
}

crf::crf(const unsigned width, const unsigned height, const unsigned labels, const std::vector<float> unary, const float lambda, const std::vector<float> pairwise)
    : width_(width)
    , height_(height)
    , labels_(labels)
    , dev_unary_(0)
    , lambda_(lambda)
    , type_(ARRAY)
    , trunc_(0)
    , dev_pairwise_(0) {

    cuda_check(cudaMalloc(&dev_unary_, width * height * labels * sizeof(float)));
    cuda_check(cudaMalloc(&dev_pairwise_, width * height * labels * labels * 2 * sizeof(float)));

    cuda_check(cudaMemcpy(dev_unary_, &unary[0], width * height * labels * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(dev_pairwise_, &pairwise[0], width * height * labels * labels * sizeof(float), cudaMemcpyHostToDevice));
}

crf::crf(const crf &prev, int)
    : width_((prev.width_ + 1) / 2)
    , height_((prev.height_ + 1) / 2)
    , labels_(prev.labels_)
    , dev_unary_(0)
    , lambda_(prev.lambda_)
    , type_(prev.type_)
    , trunc_(prev.trunc_)
    , dev_pairwise_(0) {

    if (prev.type_ == ARRAY) {
        throw std::runtime_error("Cannot generate a scaled down version of a CRF with explicit pairwise potential are specified");
    }

    cuda_check(cudaMalloc(&dev_unary_, width_ * height_ * labels_ * sizeof(float)));

    // initalise the new potential from the previous one
    dim3 block(16, 16);
    dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);

    fill_next_layer_pot<<<grid, block>>>(labels_, width_, height_, prev.width_, prev.height_, prev.dev_unary_, dev_unary_);
    cuda_check(cudaGetLastError());
}

crf::~crf() {
    cudaFree(dev_unary_);

    if (dev_pairwise_) {
        cudaFree(dev_pairwise_);
    }
}

}
}
