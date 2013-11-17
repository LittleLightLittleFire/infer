#ifndef INFER_CUDA_TRBP_H
#define INFER_CUDA_TRBP_H

#include "cuda/bp.h"

#include <vector>

namespace infer {
namespace cuda {

/**
 * Synchronous tree reweighted belief propagation on CUDA
 */
class trbp : public bp {
private:
    float *dev_rho_;

public:
    explicit trbp(const crf &crf, const std::vector<float> rho);

    virtual void run(const unsigned iterations);
    virtual void update_dev_result() const;

    virtual ~trbp();
};

}
}
#endif // INFER_CUDA_TRBP_H
