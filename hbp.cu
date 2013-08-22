#include <vector>

#include <cuda.h>

namespace {
    typedef unsigned char uchar;
    typedef unsigned uint;
}

std::vector<uchar> decode_hbp(const uchar labels, const uint layers, const uint iterations, const uint width, const uint height, const std::vector<float> &unary_psi, const float disc_trunc) {
    return std::vector<uchar>(width * height * labels);
}
