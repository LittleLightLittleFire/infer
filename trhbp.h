#ifndef HBP_H
#define HBP_H

#include <vector>

std::vector<unsigned char> decode_trhbp(const unsigned char labels, const std::vector<unsigned> &layers, const unsigned width, const unsigned height, const std::vector<float> &unary_psi, const std::vector<std::vector<float>> &rho, const float lambda, const float smooth_trunc, const bool sync);

#endif // HBP_H