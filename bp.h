#ifndef INFER_BP_H
#define INFER_BP_H

#include "method.h"

#include <vector>

namespace infer {

/**
 * Max product belief propagation
 */
class bp : public method {
protected:
    const bool synchronous_;
    unsigned current_iter;
    const indexer ndx_;

    /** Messages coming out of */
    std::vector<float> up_, down_, left_, right_;

public:
    explicit bp(const crf &crf, const bool synchronous);

    virtual void run(const unsigned iterations) override;
    virtual unsigned get_label(const unsigned x, const unsigned y) const override;

    virtual ~bp() = default;

protected:
    float *msg(std::vector<float> &msg, const unsigned x, const unsigned y) const;
    float msg(const std::vector<float> &msg, const unsigned x, const unsigned y, const unsigned label) const;
};

}

#endif // INFER_BP_H
