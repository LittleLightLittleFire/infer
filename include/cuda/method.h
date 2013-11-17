#ifndef INFER_CUDA_METHOD_H
#define INFER_CUDA_METHOD_H

#include "cuda/crf.h"

#include <vector>

namespace infer {
namespace cuda {

/**
 * Base class for CUDA inference methods
 */
class method {

public:
    const crf &crf_;

protected:
    unsigned *dev_result_;
    mutable bool dirty_; // is dev_result_ out of sync with the current gpu state

public:
    explicit method(const crf &crf);

    /**
     * Run the inference method
     * @param iterations number of iterations to run
     */
    virtual void run(const unsigned iterations) = 0;

    /**
     * Get the result of the computation
     */
    virtual std::vector<unsigned> get_result() const;

    virtual ~method();

protected:
    virtual void update_dev_result() const = 0;

private:
    // disable copy and copy assignment
    method(const method &);
    method &operator=(const method &);
};

}
}
#endif // INFER_CUDA_METHOD_H
