#include "stereo.h"
#include "iseg.h"

#include "bp.h"
#include "trbp.h"
#include "mst.h"
#include "compose.h"

#ifdef GPU_SUPPORT
#include "cuda/bp.h"
#include "cuda/trbp.h"
#endif

#include <string>
#include <iostream>
#include <stdexcept>

namespace {
    const unsigned rounds = 50;
    const unsigned mst_samples = 200;

    const unsigned hbp_layers = 5;
    const unsigned rounds_per_layer = 10;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "./driver [algorithm] [example] [...]\n"
                  << "    [algorithm] - bp, bp_async, trbp, trbp_async, hbp, hbp_async, gpu_bp, gpu_trbp, or gpu_hbp\n"
                  << "    [example] - stereo, iseg, or restore\n\n"
                  << " usage: \n"
                  << "    ./driver [algorithm] stereo [labels] [scale] [left_image.png] [right_image.png] [out_image.png]\n"
                  << "    ./driver [algorithm] iseg [image.png] [annotation.png] [out_image.png]\n"
                  << "    ./driver [algorithm] restore [image] [out_image]\n\n"
                  << "    The driver by default runs each algorithm for 50 rounds, in the case of HBP, for 5 layers with 10 rounds in each\n"
                  << std::endl;
        return 1;
    }

    const std::string algorithm = argv[1];
    const std::string example = argv[2];

    // give a CRF, return the result using the chosen algorithm
    std::function<const std::vector<unsigned>(const infer::crf)> method = [&algorithm](const infer::crf crf) {
        if (algorithm == "bp") {
            infer::bp bp(crf, true);
            bp.run(rounds);
            return bp.get_result();
        } else if (algorithm == "bp_async") {
            infer::bp bp(crf, false);
            bp.run(rounds);
            return bp.get_result();
        } else if (algorithm == "trbp") {
            infer::trbp trbp(crf, infer::sample_edge_apparence(crf.width_, crf.height_, mst_samples), true);
            trbp.run(rounds);
            return trbp.get_result();
        } else if (algorithm == "trbp_async") {
            infer::trbp trbp(crf, infer::sample_edge_apparence(crf.width_, crf.height_, mst_samples), false);
            trbp.run(rounds);
            return trbp.get_result();
        } else if (algorithm == "hbp") {
            return infer::compose<infer::bp>(hbp_layers, rounds_per_layer, crf, [](const infer::crf &downsized) { return infer::bp(downsized, true); });
        } else if (algorithm == "hbp_async") {
            return infer::compose<infer::bp>(hbp_layers, rounds_per_layer, crf, [](const infer::crf &downsized) { return infer::bp(downsized, false); });
        }

#ifdef GPU_SUPPORT
        if (algorithm == "gpu_bp") {
            infer::cuda::crf gpu_crf = crf.to_gpu();
            infer::cuda::bp(gpu_crf);
            bp.run(rounds);
            return bp.get_result();
        } else if (algorithm == "gpu_trbp") {
            infer::cuda::crf gpu_crf = crf.to_gpu();
            infer::cuda::trbp(gpu_crf, infer::sample_edge_apparence(crf.width_, crf.height_, mst_samples));
            trbp.run(rounds);
            return trbp.get_result();
        }
#else
        if (algorithm == "gpu_bp" || algorithm == "gpu_trbp") {
            throw std::runtime_error("GPU_SUPPORT was not switched on during build");
        }
#endif

        throw std::runtime_error("Unknown algorithm: " + algorithm);
    };

    if (example == "stereo") {
        if (argc < 8) {
            std::cerr << "Not enough arguments for stereo" << std::endl;
            return 1;
        }
        stereo::run(method, std::stoi(argv[3]), std::stoi(argv[4]), argv[5], argv[6], argv[7]);
    } else if (example == "iseg") {
        if (argc < 6) {
            std::cerr << "Not enough arguments for iseg" << std::endl;
            return 1;
        }
        iseg::run(method, argv[3], argv[4], argv[5]);
    } else if (example == "restore") {
    } else {
        std::cerr << "Unknown example" << std::endl;
    }

    return 0;
}
