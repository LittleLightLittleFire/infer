#ifndef INFER_TRBP_H
#define INFER_TRBP_H

#include "bp.h"
#include "util.h"

#include <vector>

namespace infer {

/**
 * Tree reweighted belief propagation
 */
class trbp : public bp {
protected:
    std::vector<float> rho_;
    edge_indexer edx_;

public:
    explicit trbp(const crf &crf, std::vector<float> rho, const bool synchronous);

    virtual void run(const unsigned iterations) override;
    virtual unsigned get_label(const unsigned x, const unsigned y) const override;

    virtual ~trbp() = default;
};

}

#endif // INFER_TRBP_H
