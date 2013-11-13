#ifndef INFER_METHOD_H
#define INFER_METHOD_H

#include "crf.h"
#include "util.h"

#include <tuple>

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
     * Get the energy of the current assignment
     */
    virtual std::tuple<float, float> get_energy() const;

    /**
     * Get the label of the specified node
     */
    virtual unsigned get_label(const unsigned x, const unsigned y) const = 0;

    virtual ~method() = default;

    // disable copy and copy assignment
    method(const method &) = delete;
    method &operator=(const method &) = delete;
};

}

#endif // INFER_METHOD_H
