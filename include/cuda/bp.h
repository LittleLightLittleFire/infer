#ifndef INFER_CUDA_BP_H
#define INFER_CUDA_BP_H

#include "cuda/method.h"

namespace infer {
namespace cuda {

/**
 * Synchronous belief propagation on CUDA
 */
class bp : public method {

public:
    explicit bp(const crf &crf);

    virtual void run(const unsigned iterations);

    virtual float unary_energy() const;
    virtual float pairwise_energy() const;

    virtual std::vector<unsigned> get_result() const;

    virtual ~bp();
};

}
}
#endif // INFER_CUDA_BP_H
