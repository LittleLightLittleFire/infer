#ifndef INFER_COMPOSE_H
#define INFER_COMPOSE_H

#include "crf.h"
#include "method.h"
#include "util.h"

#include <functional>
#include <iostream>

namespace infer {

/**
 * Generalised hierarchical belief propagation
 * Works with any class that has a constructor of the form M(crf &, M)
 */
template <class M>
std::vector<unsigned> compose(const unsigned layers, const unsigned iterations, const crf &crf, const std::function<M(const infer::crf &)> init, const std::function<M(int, const infer::crf &, const M &)> get) {
    // generate potentials for the layers
    std::vector<infer::crf> crfs;

    for (unsigned i = 0; i < layers - 1; ++i) {
        crfs.push_back(i == 0 ? crf.downsize() : crfs[i-1].downsize());
    }

    // recursive function to solve all of our move constructor problems
    std::function<M(const int, M)> func = [&crf, &crfs, &func, &get, iterations](const int i, M m) {
        m.run(iterations);

        if (i > 0) {
            return func(i - 1, get(i, crfs[i - 1], m));
        } else if (i == 0) {
            return func(-1, get(0, crf, m));
        } else {
            return m;
        }
    };

    return func(crfs.size() - 1, init(crfs.back())).get_result();
}

std::vector<unsigned> hbp(const unsigned layers, const unsigned iterations, const bool sync, const crf &crf) {
    return compose<bp>(layers, iterations, crf, [sync](const infer::crf &downsized) {
        return bp(downsized, sync);
    }, [](int, const infer::crf &new_crf, const bp &prev) {
        return bp(new_crf, prev);
    });
}

std::vector<unsigned> trhbp(const unsigned layers, const unsigned iterations, const std::vector<std::vector<float>> rho, const bool sync, const crf &crf) {
    return compose<trbp>(layers, iterations, crf, [sync, &rho](const infer::crf &downsized) {
        return trbp(downsized, rho.back(), sync);
    }, [&rho](int round, const infer::crf &new_crf, const trbp &prev) {
        return trbp(new_crf, rho[round], prev);
    });
}

}

#endif // INFER_COMPOSE_H
