#include "cuda/core.h"

#include <cuda.h>
#include <math_constants.h>

namespace infer {
namespace cuda {

namespace {
__device__ void fast_L1_helper(const unsigned labels, const float curr_min, const float s, const float t, float *out) {

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
    const float val = out[0];
    for (unsigned i = 0; i < labels; ++i) {
        out[i] -= val;
    }
}

__device__ void send_message_L1(const unsigned labels, const unsigned x, const unsigned y, const float lambda, const float trunc, const float *m1, const float *m2, const float *m3, const float *pot, float *out) {
    float curr_min = CUDART_MAX_NORMAL_F;

    // add all the incoming messages together
    for (unsigned i = 0; i < labels; ++i) {
        out[i] = m1[i] + m2[i] + m3[i] + pot[i];
        curr_min = fminf(curr_min, out[i]);
    }

    // adjust lambda because of trbp
    fast_L1_helper(labels, curr_min, lambda, lambda * trunc, out);
}

__device__ void send_message_L1(const unsigned labels, const unsigned x, const unsigned y, const float lambda, const float trunc
                               , const float *m1, const float *m2, const float *m3, const float *opp
                               , const float rm1, const float rm2, const float rm3, const float ropp
                               , const float *pot, float *out) {

    float curr_min = CUDART_MAX_NORMAL_F;

    // add all the incoming messages together
    for (unsigned i = 0; i < labels; ++i) {
        out[i] = pot[i] + m1[i] * rm1 + m2[i] * rm2 + m3[i] * rm3 - opp[i] * (1 - ropp);
        curr_min = fminf(curr_min, out[i]);
    }

    // adjust lambda because of trbp
    fast_L1_helper(labels, curr_min, lambda * (1 / ropp), lambda * trunc * (1 / ropp), out);
}

__device__ float *ndx(const unsigned labels, const unsigned width, float *dir, const unsigned x, const unsigned y) {
    return labels * (x + y * width) + dir;
}

__device__ const float *cndx(const unsigned labels, const unsigned width, const float *dir, const unsigned x, const unsigned y) {
    return labels * (x + y * width) + dir;
}

}

/** initalise messages using the messages from the layer below */
__global__ void prime(const unsigned lbl, const unsigned w, const unsigned h, const unsigned prev_w, const float *prev_msg, float *out) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    // boundary check
    if (x >= w || y >= h) {
        return;
    }

    // initaise to the last layer's (x/2, y/2)
    float *target = ndx(lbl, w, out, x, y);
    const float *source = cndx(lbl, prev_w, prev_msg, x / 2, y / 2);

    for (unsigned i = 0; i < lbl; ++i) {
        target[i] = source[i];
    }
}


/** generate the next layer's potentials */
__global__ void fill_next_layer_pot(const unsigned labels, const unsigned width, const unsigned height, const unsigned max_width, const unsigned max_height, const float *pot, float *out) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    // bounds check
    if (x >= width || y >= height) {
        return;
    }

    // collapse the potential in a 2x2 area
    float *target = ndx(labels, width, out, x, y);
    const float *top_left = cndx(labels, max_width, pot, 2 * x, 2 * y);;

    for (unsigned i = 0; i < labels; ++i) {
        target[i] = top_left[i];
    }

    if (2 * x + 1 < max_width) {
        const float *top_right = cndx(labels, max_width, pot, 2 * x + 1, 2 * y);;
        for (unsigned i = 0; i < labels; ++i) {
            target[i] += top_right[i];
        }
    }

    if (2 * (y + 1) < max_height) {
        const float *bottom_left = cndx(labels, max_width, pot, 2 * x, 2 * (y + 1));
        for (unsigned i = 0; i < labels; ++i) {
            target[i] += bottom_left[i];
        }
    }

    if (2 * x + 1 < max_width && 2 * (y + 1) < max_height) {
        const float *bottom_right = cndx(labels, max_width, pot, 2 * x + 1, 2 * (y + 1));
        for (unsigned i = 0; i < labels; ++i) {
            target[i] += bottom_right[i];
        }
    }
}

__global__ void trbp_run(const unsigned labels, const unsigned w, const unsigned h, const unsigned i, const crf::type type, const float lambda, const float trunc, const float *pairwise, float *l, float *r, float *u, float *d, const float *pot, const float *rho) {
    const unsigned ix = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned iy = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned x = ix * 2 + ((i + iy) % 2 == 0 ? 1 : 0);
    const unsigned y = iy;

    // bounds check
    if (x == 0 || y == 0 || x >= w - 1 || y >= h - 1) {
        return;
    }

    //printf("thread (%u, %u), block (%u, %u) %u %u\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, x, y);

    const float up = rho[(w * y + x) * 2];
    const float left = rho[(w * y + x) * 2 + 1];
    const float down = rho[(w * (y - 1) + x) * 2];
    const float right = rho[(w * y + (x - 1)) * 2 + 1];

    const unsigned base = (w * y + x) * labels;
    switch (type) {
        case crf::L1:
            if (rho) {
                //                                           m1        m2        m3        opp
                send_message_L1(labels, x, y, lambda, trunc, u + base, d + base, r + base, l + base
                                                           , up      , down    , right   , left     , pot + base, r + (w * y + x + 1) * labels);

                send_message_L1(labels, x, y, lambda, trunc, u + base, d + base, l + base, r + base
                                                           , up      , down    , left    , right    , pot + base, l + (w * y + x - 1) * labels);

                send_message_L1(labels, x, y, lambda, trunc, d + base, l + base, r + base, u + base
                                                           , down    , left    , right   , up       , pot + base, d + (w * (y + 1) + x) * labels);

                send_message_L1(labels, x, y, lambda, trunc, u + base, l + base, r + base, d + base
                                                           , up      , left    , right   , down     , pot + base, u + (w * (y - 1) + x) * labels);
            } else {
                send_message_L1(labels, x, y, lambda, trunc, u + base, d + base, r + base, pot + base, r + (w * y + x + 1) * labels);
                send_message_L1(labels, x, y, lambda, trunc, u + base, d + base, l + base, pot + base, l + (w * y + x - 1) * labels);
                send_message_L1(labels, x, y, lambda, trunc, d + base, l + base, r + base, pot + base, d + (w * (y + 1) + x) * labels);
                send_message_L1(labels, x, y, lambda, trunc, u + base, l + base, r + base, pot + base, u + (w * (y - 1) + x) * labels);
            }
            break;
        case crf::L2: // TODO:
            break;
        case crf::ARRAY: // TODO:
            break;
    }
}

__global__ void trbp_get_results(const unsigned labels, const unsigned w, const unsigned h, const float *l, const float *r, const float *u, const float *d, const float *pot, unsigned *out, const float *rho) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    // bounds check
    if (x >= w || y >= h) {
        return;
    }

    unsigned min_label = 0;
    float min_value = CUDART_MAX_NORMAL_F;

    const unsigned base = (w * y + x) * labels;

    for (unsigned i = 0; i < labels; ++i) {
        float val = (pot + base)[i];

        if (rho) {
            if (y != h - 1) val += (u + base)[i] * rho[(w * y + x) * 2];
            if (x != w - 1) val += (l + base)[i] * rho[(w * y + x) * 2 + 1];
            if (y != 0)     val += (d + base)[i] * rho[(w * (y - 1) + x) * 2];
            if (x != 0)     val += (r + base)[i] * rho[(w * y + (x - 1)) * 2 + 1];
        } else {
            val += (l + base)[i] + (r + base)[i] + (u + base)[i] + (d + base)[i];
        }

        if (val < min_value) {
            min_label = i;
            min_value = val;
        }
    }

    out[x + y * w] = min_label;
}

}
}
