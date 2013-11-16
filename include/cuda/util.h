#ifndef INFER_CUDA_UTIL_H
#define INFER_CUDA_UTIL_H

#include <cuda.h>

#include <string>
#include <stdexcept>

#define cuda_check(result) { cuda_throw(__FILE__, __LINE__, (result)); }

namespace infer {
namespace cuda {

class cuda_exception : public std::runtime_error {
public:
    explicit cuda_exception(std::string err);
};

void cuda_throw(const char *file, const unsigned line, const cudaError_t err);

}
}

#endif // INFER_CUDA_UTIL_H
