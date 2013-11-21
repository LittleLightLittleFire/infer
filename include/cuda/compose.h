#ifndef INFER_CUDA_COMPOSE_H
#define INFER_CUDA_COMPOSE_H

#include "cuda/crf.h"

namespace infer {
namespace cuda {

std::vector<unsigned> hbp(const unsigned layers, const unsigned iterations, const gpu_crf &crf);

}
}

#endif // INFER_CUDA_COMPOSE_H
