#ifndef HBP_H
#define HBP_H

#include <vector>

std::vector<unsigned char> decode_trhbp(const unsigned char labels, const unsigned layers, const unsigned iterations, const unsigned width, const unsigned height, const std::vector<float> &unary_psi, const std::vector<std::vector<float>> &rho, const float disc_trunc);

#endif // HBP_H
