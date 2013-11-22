#ifndef INFER_CUDA_QP_H
#define INFER_CUDA_QP_H

#include "cuda/method.h"
#include <string>

namespace infer {
namespace cuda {

/**
 * Quadratic programming relaxation on CUDA
 */
class qp : public method {
protected:
    float *dev_mu1_, *dev_mu2_, *dev_pair_;

public:
    explicit qp(const crf &crf);

    virtual void run(const unsigned iterations);
    virtual void update_dev_result() const;
    virtual std::string get_name() const;
    virtual ~qp();
};

}
}
#endif // INFER_CUDA_QP_H
