#include "cuda/compose.h"
#include "cuda/bp.h"

namespace infer {
namespace cuda {

std::vector<unsigned> hbp(const unsigned layers, const unsigned iterations, const crf &crf) {
    // we'll be breaking RAII here, since CUDA doesn't yet support C++11 and it is not
    // worth the effort to hand roll a unique_ptr or use boost
    // also there are no move semantics

    // what this means is that if we hit an exception we'll leak a lot of memory

    // generate a stack of CRFs
    std::vector<const cuda::crf *> crfs;
    for (unsigned i = 0; i < layers; ++i) {
        crfs.push_back(i == 0 ? &crf : new cuda::crf(*crfs[i - 1], 0));
    }

    // create the method on the smallest layer
    bp *m = new bp(*crfs.back());
    m->run(iterations);

    for (unsigned i = layers - 1; i-- > 0; ) {
        bp *const new_m = new bp(*crfs[i], *m);
        delete m;

        m = new_m;
        m->run(iterations);
    }

    // get the results
    std::vector<unsigned> result = m->get_result();
    delete m;

    // clean up
    for (unsigned i = 1; i < crfs.size(); ++i) {
        delete crfs[i];
    }

    return result;
}

}
}
