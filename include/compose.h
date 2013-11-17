#ifndef INFER_COMPOSE_H
#define INFER_COMPOSE_H

#include "crf.h"
#include "method.h"
#include "util.h"

#include <functional>

namespace infer {

/**
 * Generalised hierarchical belief propagation
 * Works with any class that has a constructor of the form M(crf &, M)
 */
template <class M>
std::vector<unsigned> compose(const unsigned layers, const unsigned iterations, const crf &crf, const std::function<M(const infer::crf &)> init) {
    // generate potentials for the layers
    std::vector<infer::crf> crfs;

    for (unsigned i = 0; i < layers - 1; ++i) {
        crfs.push_back(i == 0 ? crf.downsize() : crfs[i-1].downsize());
    }

    // recursive function to solve all of our move constructor problems
    std::function<M(const int, M)> func = [&crf, &crfs, &func, iterations](const int i, M m) {
        m.run(iterations);

        if (i > 0) {
            return func(i - 1, M(crfs[i - 1], m));
        } else if (i == 0) {
            return func(-1, M(crf, m));
        } else {
            return m;
        }
    };

    return func(crfs.size() - 1, init(crfs.back())).get_result();
}

}

#endif // INFER_COMPOSE_H
