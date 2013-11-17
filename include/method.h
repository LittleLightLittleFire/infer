#ifndef INFER_METHOD_H
#define INFER_METHOD_H

#include "crf.h"

namespace infer {

/**
 * Base class for inference methods
 */
class method {

public:
    const crf &crf_;

    /**
     * Initalise the inference method with conditional random field
     */
    explicit method(const crf &crf);

    /**
     * Run the inference method
     * @param iterations number of iterations to run
     */
    virtual void run(const unsigned iterations) = 0;

    /**
     * Get the label of the specified node
     */
    virtual unsigned get_label(const unsigned x, const unsigned y) const = 0;

    /**
     * Get results, equivalent to calling get_label for every pixel
     */
    virtual std::vector<unsigned> get_result() const;

    /**
     * TODO: change back to default when gcc is updated, gcc 4.7 has a bug with virtual destructors being defaulted
     */
    virtual ~method();

    // disable copy and copy assignment
    method(const method &) = delete;
    method &operator=(const method &) = delete;
};

}

#endif // INFER_METHOD_H
