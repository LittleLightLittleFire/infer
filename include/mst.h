#ifndef INFER_MST_H
#define INFER_MST_H

#include <vector>

namespace infer {
std::vector<float> sample_edge_apparence(const unsigned width, const unsigned height, const unsigned max_iter);
std::vector<std::vector<float>> sample_edge_apparence(unsigned width, unsigned height, const unsigned max_iter, const unsigned layers);
}

#endif // INFER_MST_H
