#include "cuda/core.h"

#include <cuda.h>
#include <math_constants.h>

namespace infer {
namespace cuda {

namespace {
__device__ float *ndx(const unsigned labels, const unsigned width, float *dir, const unsigned x, const unsigned y) {
    return labels * (x + y * width) + dir;
}

__device__ const float *cndx(const unsigned labels, const unsigned width, const float *dir, const unsigned x, const unsigned y) {
    return labels * (x + y * width) + dir;
}

__device__ const float *edx(const unsigned labels, const unsigned width, const float *pairwise, const unsigned rdx) {
    return width * labels * labels + pairwise;
}

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

__device__ void send_message_array(const unsigned labels, const unsigned x, const unsigned y, const float lambda, const float *edge_pot, const float *m1, const float *m2, const float *m3, const float *pot, float *out) {
    for (unsigned j = 0; j < labels; ++j) {
        out[j] = CUDART_MAX_NORMAL_F;
        for (unsigned i = 0; i < labels; ++i) {
            const float pairwise = lambda * edge_pot[j * labels + i];
            const float unary = m1[i] + m2[i] + m3[i] + pot[i];

            out[j] = fminf(out[j], unary + pairwise);
        }
    }
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

    const float down = (w * y + x) * 2;
    const float right = (w * y + x) * 2 + 1;
    const float up = (w * (y - 1) + x) * 2;
    const float left = (w * y + (x - 1)) * 2 + 1;

    const unsigned base = (w * y + x) * labels;
    if (rho) {
        // directions are reversed since the edges are pointing into the node
        const float up_ = rho[down]
        const float left_ = rho[right];
        const float down_ = rho[up];
        const float right_ = rho[left];

        switch (type) {
            case crf::L1:
                //                                           m1        m2        m3        opp
                send_message_L1(labels, x, y, lambda, trunc, u + base, d + base, r + base, l + base
                                                           , up_     , down_   , right_  , left_    , pot + base, r + (w * y + x + 1) * labels);

                send_message_L1(labels, x, y, lambda, trunc, u + base, d + base, l + base, r + base
                                                           , up_     , down_   , left_   , right_   , pot + base, l + (w * y + x - 1) * labels);

                send_message_L1(labels, x, y, lambda, trunc, d + base, l + base, r + base, u + base
                                                           , down_   , left_   , right_  , up_      , pot + base, d + (w * (y + 1) + x) * labels);

                send_message_L1(labels, x, y, lambda, trunc, u + base, l + base, r + base, d + base
                                                           , up_     , left_   , right_  , down_    , pot + base, u + (w * (y - 1) + x) * labels);
                break;
            case crf::L2: // TODO:
                break;
            case crf::ARRAY: // TODO:
                break;
        }
    } else {
        switch (type) {
            case crf::L1:
                send_message_L1(labels, x, y, lambda, trunc, u + base, d + base, r + base, pot + base, r + (w * y + x + 1) * labels);
                send_message_L1(labels, x, y, lambda, trunc, u + base, d + base, l + base, pot + base, l + (w * y + x - 1) * labels);
                send_message_L1(labels, x, y, lambda, trunc, d + base, l + base, r + base, pot + base, d + (w * (y + 1) + x) * labels);
                send_message_L1(labels, x, y, lambda, trunc, u + base, l + base, r + base, pot + base, u + (w * (y - 1) + x) * labels);
                break;
            case crf::L2: // TODO:
                break;
            case crf::ARRAY:
                send_message_array(labels, x, y, lambda, edx(labels, w, pairwise, right), u + base, d + base, r + base, pot + base, r + (w * y + x + 1) * labels);
                send_message_array(labels, x, y, lambda, edx(labels, w, pairwise, left) , u + base, d + base, l + base, pot + base, l + (w * y + x - 1) * labels);
                send_message_array(labels, x, y, lambda, edx(labels, w, pairwise, down) , d + base, l + base, r + base, pot + base, d + (w * (y + 1) + x) * labels);
                send_message_array(labels, x, y, lambda, edx(labels, w, pairwise, up)   , u + base, l + base, r + base, pot + base, u + (w * (y - 1) + x) * labels);
                break;
        }
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
