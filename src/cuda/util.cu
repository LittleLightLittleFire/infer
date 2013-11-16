#include "cuda/util.h"

#include <sstream>

namespace infer {
namespace cuda {

cuda_exception::cuda_exception(std::string err)
    : runtime_error(err) {

}

void cuda_throw(const char *file, const unsigned line, const cudaError_t err) {
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << file << ":" << line <<  " " << cudaGetErrorString(err);
        throw cuda_exception(ss.str());
    }
}

}
}

