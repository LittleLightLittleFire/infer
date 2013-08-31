#ifndef HBP_H
#define HBP_H

#include <vector>

std::vector<unsigned char> decode_hbp(const unsigned char labels, const unsigned layers, const unsigned iterations, const unsigned width, const unsigned height, const std::vector<float> &unary_psi, const float disc_trunc);

#endif // HBP_H
