#include "cuda/bp.h"
#include "cuda/util.h"

#include <cuda.h>
#include <math_constants.h>

namespace infer {
namespace cuda {

namespace {

__device__ void send_message_L1(const unsigned labels, const unsigned x, const unsigned y, const float lambda, const float trunc, const float *m1, const float *m2, const float *m3, const float *pot, float *out) {
    float curr_min = CUDART_MAX_NORMAL_F;

    // add all the incoming messages together
    for (unsigned i = 0; i < labels; ++i) {
        out[i] = m1[i] + m2[i] + m3[i] + pot[i];
        curr_min = fminf(curr_min, out[i]);
    }

    // adjust lambda because of trbp
    const float s = lambda;
    const float t = lambda * trunc;

    // do the O(n) trick
    for (unsigned i = 1; i < labels; ++i) {
        out[i] = fminf(out[i-1] + s, out[i]);
    }

    for (unsigned i = labels - 1; i-- > 0; ) {
        out[i] = fminf(out[i+1] + s, out[i]);
    }

    // truncate
    for (unsigned i = 0; i < labels; ++i) {
        out[i] = fminf(curr_min + t, out[i]);
    }

    // normalise
    /*
    float sum = 0;
    for (unsigned i = 0; i < labels; ++i) {
        sum += out[i];
    }
    sum /= static_cast<float>(labels);
    */

    const float val = out[0];
    for (unsigned i = 0; i < labels; ++i) {
        out[i] -= val;
    }
}

__global__ void bp_run(const unsigned labels, const unsigned w, const unsigned h, const unsigned i, const crf::type type, const float lambda, const float trunc, const float *pairwise, float *l, float *r, float *u, float *d, float *pot) {
    const unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned x = ix * 2 + ((i + iy) % 2 == 0 ? 1 : 0);
    const unsigned y = iy;

    // bounds check
    if (x >= w || y >= h) {
        return;
    }

    const unsigned base = (w * y + x) * labels;
    switch (type) {
        case crf::L1:
            send_message_L1(labels, x, y, lambda, trunc, u + base, d + base, r + base, pot + base, r + (w * y + x + 1) * labels);
            send_message_L1(labels, x, y, lambda, trunc, u + base, d + base, l + base, pot + base, l + (w * y + x - 1) * labels);
            send_message_L1(labels, x, y, lambda, trunc, d + base, l + base, r + base, pot + base, d + (w * (y + 1) + x) * labels);
            send_message_L1(labels, x, y, lambda, trunc, u + base, l + base, r + base, pot + base, u + (w * (y - 1) + x) * labels);
            break;
        case crf::L2: // TODO:
            break;
        case crf::ARRAY: // TODO:
            break;
    }
}

__global__ void bp_get_results(const unsigned labels, const unsigned w, const unsigned h, const float *l, const float *r, const float *u, const float *d, const float *pot, unsigned *out) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // bounds check
    if (x >= w || y >= h) {
        return;
    }

    unsigned min_label = 0;
    float min_value = CUDART_MAX_NORMAL_F;

    const unsigned base = (w * y + x) * labels;

    for (uint i = 0; i < labels; ++i) {
        const float val = (l + base)[i]
                        + (r + base)[i]
                        + (u + base)[i]
                        + (d + base)[i]
                        + (pot + base)[i];

        if (val < min_value) {
            min_label = i;
            min_value = val;
        }
    }

    out[x + y * w] = min_label;
}

}

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
    dim3 block(8, 16); // only half of the pixels are updated because of the checkboard pattern
    dim3 grid(((crf_.width_ + 1) / 2 + block.x - 1) / block.x, (crf_.height_ + block.y - 1) / block.y);

    for (unsigned i = 0; i < iterations; ++i) {
        ++current_iteration;

        bp_run<<<grid, block>>>(crf_.labels_, crf_.width_, crf_.height_, current_iteration, crf_.type_, crf_.lambda_, crf_.trunc_, crf_.dev_pairwise_, dev_l_, dev_r_, dev_u_, dev_d_, crf_.dev_unary_);
    }
}

void bp::update_dev_result() const {
    dim3 block(16, 16);
    dim3 grid((crf_.width_ + block.x - 1) / block.x, (crf_.height_ + block.y - 1) / block.y);
    bp_get_results<<<grid, block>>>(crf_.labels_, crf_.width_, crf_.height_, dev_l_, dev_r_, dev_u_, dev_d_, crf_.dev_unary_, dev_result_);
}

bp::~bp() {
    cudaFree(dev_l_);
    cudaFree(dev_r_);
    cudaFree(dev_u_);
    cudaFree(dev_d_);
}

}
}
