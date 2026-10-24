#ifndef LSTM_FORWARD_H
#define LSTM_FORWARD_H

#include <ap_fixed.h>
#include <hls_math.h>


typedef ap_fixed<8, 4> slope_t;

typedef ap_fixed<8, 2> weight_t;
typedef ap_fixed<8, 2> bias_t;
typedef ap_fixed<16, 6> data_t;
typedef ap_fixed<32, 12> acc_t;
typedef ap_fixed<28, 10> acc_t_fc;

#endif
