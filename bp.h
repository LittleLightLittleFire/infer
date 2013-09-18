#ifndef BP_H
#define BP_H

#include <vector>
#include <functional>

/** A struct that packs in all the data required by a send_message function */
struct message_data {
    const float *m1, *m2, *m3, *opp, *pot; // 3 messages in and the opposite message
    float *out;

    const float rm1, rm2, rm3, ropp;
    unsigned labels;  // put this last for alignment on 64 bit systems
};

void send_msg_lin_trunc(const message_data in, const float lambda, const float data_disc);

std::vector<unsigned char> decode_trbp(const uint labels, const uint max_iter, const uint width, const uint height, const std::vector<float> &pot, const std::vector<float> &rho, const std::function<void(message_data)> send_msg, const bool sync);

#endif // BP_H
