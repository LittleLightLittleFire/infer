#ifndef INFER_BP_H
#define INFER_BP_H

#include "method.h"

#include <string>
#include <vector>

namespace infer {

/**
 * Belief propagation
 */
class bp : public method {
protected:
    const bool synchronous_;
    unsigned current_iter;
    const indexer ndx_;

    /**
     * Messages coming into the node
     *
     * Draw a dot and draw four arrows coming into the dot, one each of the neighbours
     * The arrow pointing up is in up_, the arrow pointing down is in down_, etc etc
     */
    std::vector<float> up_, down_, left_, right_;

public:
    explicit bp(const crf &crf, const bool synchronous);
    explicit bp(const crf &crf, const bp &prev);

    virtual void run(const unsigned iterations) override;
    virtual unsigned get_label(const unsigned x, const unsigned y) const override;
    virtual std::string get_name() const override;

    virtual ~bp() = default;
    bp(bp &&) = default;

protected:
    float *msg(std::vector<float> &msg, const unsigned x, const unsigned y) const;
    float msg(const std::vector<float> &msg, const unsigned x, const unsigned y, const unsigned label) const;

};

}

#endif // INFER_BP_H
