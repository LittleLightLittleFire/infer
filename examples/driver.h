#ifndef DRIVER_H
#define DRIVER_H

#include "crf.h"
#include <functional>

#include <string>

namespace stereo {
void run(const std::function<const std::vector<unsigned>(const infer::crf)> method, const unsigned labels, const unsigned scale, const std::string left_name, const std::string right_name, const std::string output_name);
}

namespace iseg {
void run(const std::function<const std::vector<unsigned>(const infer::crf)> method, const std::string image_name, const std::string anno_name, const std::string output_name);
}

#endif // DRIVER_H
