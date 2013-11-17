#include "cuda/bp.h"
#include "cuda/util.h"

#include <cuda.h>

namespace infer {
namespace cuda {

namespace {

__device__ void send_message_L1(const unsigned lbl, const unsigned x, const unsigned y, const float lambda, const float trunc, const float *m1, const float *m2, const float *m3, const float *pot, float *out) {
}

__global__ void run_bp(const unsigned lbl, const unsigned w, const unsigned h, const unsigned i, const crf::type type, const float lambda, const float trunc, const float *pairwise, float *l, float *r, float *u, float *d, float *pot) {
    const unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned x = ix * 2 + ((i + iy) % 2 == 0 ? 1 : 0);
    const unsigned y = iy;

    // bounds check
    if (x >= w || y >= h) {
        return;
    }

    const unsigned base = (w * y + x) * lbl;
    switch (type) {
        case crf::L1:
            send_message_L1(lbl, x, y, lambda, trunc, u + base, d + base, r + base, pot + base, r + (w * y + x + 1) * lbl);
            send_message_L1(lbl, x, y, lambda, trunc, u + base, d + base, l + base, pot + base, l + (w * y + x - 1) * lbl);
            send_message_L1(lbl, x, y, lambda, trunc, d + base, l + base, r + base, pot + base, d + (w * (y + 1) + x) * lbl);
            send_message_L1(lbl, x, y, lambda, trunc, u + base, l + base, r + base, pot + base, u + (w * (y - 1) + x) * lbl);
            break;
        case crf::L2: // TODO:
            break;
        case crf::ARRAY: // TODO:
            break;
    }
}

}

bp::bp(const crf &crf)
    : method(crf)
    , current_iteration(0)
    , dev_l_(0), dev_r_(0), dev_u_(0), dev_d_(0) {

    const size_t size = crf.width_ * crf.height_ * crf.labels_;
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
    dim3 block(8, 16); // only half of the pixels are updated because of the checkboard pattern
    dim3 grid(((crf_.width_ + 1) / 2 + block.x - 1) / block.x, (crf_.height_ + block.y - 1) / block.y);

    for (unsigned i = 0; i < iterations; ++i) {
        ++current_iteration;

        run_bp<<<grid, block>>>(crf_.labels_, crf_.width_, crf_.height_, current_iteration, crf_.type_, crf_.lambda_, crf_.trunc_, crf_.dev_pairwise_, dev_l_, dev_r_, dev_u_, dev_d_, crf_.dev_unary_);
    }
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
    cudaFree(dev_l_);
    cudaFree(dev_r_);
    cudaFree(dev_u_);
    cudaFree(dev_d_);
}

}
}
