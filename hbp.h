#ifndef HBP_H
#define HBP_H

std::vector<unsigned char> decode_hbp(const unsigned char labels, const unsigned layers, const unsigned iterations, const unsigned width, const unsigned height, const float *dev_unary_psi, const float disc_trunc);

#endif // HBP_H
