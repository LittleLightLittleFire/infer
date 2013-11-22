#ifndef INFER_CUDA_CORE_H
#define INFER_CUDA_CORE_H

#include "cuda/crf.h"

namespace infer {
namespace cuda {

__global__ void trbp_run(const unsigned labels, const unsigned w, const unsigned h, const unsigned i, const crf::type type, const float lambda, const float trunc, const float *pairwise, float *l, float *r, float *u, float *d, const float *pot, const float *rho = 0);

__global__ void trbp_get_results(const unsigned labels, const unsigned w, const unsigned h, const float *l, const float *r, const float *u, const float *d, const float *pot, unsigned *out, const float *rho = 0);

__global__ void prime(const unsigned lbl, const unsigned w, const unsigned h, const unsigned prev_w, const float *prev_msg, float *out);

__global__ void fill_next_layer_pot(const unsigned labels, const unsigned width, const unsigned height, const unsigned max_width, const unsigned max_height, const float *pot, float *out);

__global__ void qp_initalise(const unsigned labels, const unsigned w, const unsigned h, const float *in, float *out);

__global__ void qp_run(const unsigned labels, const unsigned w, const unsigned h, const float *pot, const float lambda, const float *pairwise, const crf::type type, const float *mu, float *mu_out);

__global__ void qp_get_results(const unsigned labels, const unsigned w, const unsigned h, const float *mu, unsigned *out);

}
}

#endif // INFER_CUDA_CORE_H
