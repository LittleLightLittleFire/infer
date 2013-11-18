#ifndef INFER_QP_H
#define INFER_QP_H

#include "method.h"

#include <vector>

namespace infer {

/**
 * Quadratic programming solver
 */
class qp : public method {
private:
    const indexer ndx_;

    std::vector<float> q_;
    std::vector<float> mu1_, mu2_;
    float *mu_, *mu_next_;

public:
    explicit qp(const crf &crf);

    virtual void run(const unsigned iterations) override;
    virtual unsigned get_label(const unsigned x, const unsigned y) const override;
    virtual float objective() const;

    virtual ~qp() = default;
};

}

#endif // INFER_QP_H
