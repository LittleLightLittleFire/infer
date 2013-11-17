#ifndef INFER_CUDA_CORE_H
#define INFER_CUDA_CORE_H

#include <cuda.h>
#include <math_constants.h>

namespace infer {
namespace cuda {

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
    if (x == 0 || y == 0 || x >= w - 1 || y >= h - 1) {
        return;
    }

    //printf("thread (%u, %u), block (%u, %u) %u %u\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, x, y);

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
}

#endif // INFER_CUDA_CORE_H
