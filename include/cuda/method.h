#ifndef INFER_CUDA_METHOD_H
#define INFER_CUDA_METHOD_H

#include "cuda/crf.h"

namespace infer {
namespace cuda {

/**
 * Base class for CUDA inference methods
 */
class method {

public:
    const crf &crf_;

    explicit method(const crf &crf);

    /**
     * Run the inference method
     * @param iterations number of iterations to run
     */
    virtual void run(const unsigned iterations) = 0;

    /**
     * Get the unary energy of the current assignment
     */
    virtual float unary_energy() const = 0;

    /**
     * Get the pairwise energy of the current assignment
     */
    virtual float pairwise_energy() const = 0;

    /**
     * Get the result of the computation
     */
    virtual std::vector<unsigned> get_result() const = 0;

    virtual ~method();

    // disable copy and copy assignment
private:
    method(const method &);
    method &operator=(const method &);
};

}
}
#endif // INFER_CUDA_METHOD_H
