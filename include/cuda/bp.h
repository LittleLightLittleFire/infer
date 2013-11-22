#ifndef INFER_CUDA_BP_H
#define INFER_CUDA_BP_H

#include "cuda/method.h"
#include <string>

namespace infer {
namespace cuda {

/**
 * Synchronous belief propagation on CUDA
 */
class bp : public method {
protected:
    float *dev_l_, *dev_r_, *dev_u_, *dev_d_;
    unsigned current_iteration_;

public:
    explicit bp(const crf &crf);
    explicit bp(const crf &new_crf, const bp &prev);

    virtual void run(const unsigned iterations);
    virtual void update_dev_result() const;
    virtual std::string get_name() const;
    virtual ~bp();
};

}
}
#endif // INFER_CUDA_BP_H
