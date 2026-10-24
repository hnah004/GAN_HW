#include "lstm_forward.h"
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <hls_math.h>

#include "W_l0_fixed.h"
#include "B_l0_fixed.h"
#include "W_l1_fixed.h"
#include "B_l1_fixed.h"
#include "W_fc1_fixed.h"
#include "b_fc1_fixed.h"
#include "W_fc2_fixed.h"
#include "b_fc2_fixed.h"
#include "slope_fixed.h"
#include "tanh_lut.h"
#include "sigmoid_lut.h"

typedef ap_axiu<16, 1, 1, 1> axis_t;

// Sigmoid function
static data_t sigmoid_fast(acc_t x) {
#pragma HLS INLINE
    const acc_t MIN = (acc_t)-8.0;
    const acc_t MAX = (acc_t)+8.0;
    if (x <= MIN) return sigmoid_lut[0];
    if (x >= MAX) return sigmoid_lut[SIGMOID_LUT_SIZE-1];

    acc_t s_f = (x - MIN) * (acc_t)32;
    int idx = (int)s_f.to_int();

    if (idx < 0) idx = 0;
    if (idx > SIGMOID_LUT_SIZE - 2) idx = SIGMOID_LUT_SIZE - 2;

    acc_t frac = s_f - (acc_t)idx;
    acc_t y0 = (acc_t)sigmoid_lut[idx];
    acc_t y1 = (acc_t)sigmoid_lut[idx+1];
    return (data_t)(y0 + frac * (y1 - y0));
}

// Tanh function
static data_t tanh_fast(acc_t x) {
#pragma HLS INLINE
    const acc_t MIN = (acc_t)-4.0;
    const acc_t MAX = (acc_t)+4.0;
    if (x <= MIN) return (data_t)-1;
    if (x >= MAX) return (data_t)+1;

    acc_t s_f = (x - MIN) * (acc_t)64;
    int idx = (int)s_f.to_int();

    if (idx < 0) idx = 0;
    if (idx > TANH_LUT_SIZE - 2) idx = TANH_LUT_SIZE - 2;

    acc_t frac = s_f - (acc_t)idx;
    acc_t y0 = (acc_t)tanh_lut[idx];
    acc_t y1 = (acc_t)tanh_lut[idx+1];
    return (data_t)(y0 + frac * (y1 - y0));
}

// lstm_cell
template<int IN_SZ, int W_COL>
void lstm_cell_opt(
    data_t x_t[IN_SZ], data_t h_idx[128], data_t c_idx[128],
    data_t h_odx[128], data_t c_odx[128],
    const weight_t W[512][W_COL], const bias_t B[512]
) {
#pragma HLS INLINE off
    acc_t gates[512];

    acc_t combined[W_COL];
    #pragma HLS ARRAY_PARTITION variable=combined complete

    for (int j = 0; j < IN_SZ; j++) {
        #pragma HLS UNROLL
        combined[j] = (acc_t)x_t[j];
    }
    for (int j = 0; j < 128; j++) {
        #pragma HLS UNROLL
        combined[IN_SZ + j] = (acc_t)h_idx[j];
    }

    GEMV_OUTER: for (int i = 0; i < 512; i++) {
        acc_t acc = (acc_t)B[i];
        GEMV_INNER: for (int j = 0; j < W_COL; j++) {
            #pragma HLS PIPELINE II=1
            acc += combined[j] * (acc_t)W[i][j];
        }
        gates[i] = acc;
    }

    STATE_UPDATE: for (int i = 0; i < 128; i++) {
#pragma HLS PIPELINE II=4
        data_t i_g = sigmoid_fast(gates[i]);
        data_t f_g = sigmoid_fast(gates[i + 128]);
        data_t g_g = tanh_fast(gates[i + 256]);
        data_t o_g = sigmoid_fast(gates[i + 384]);


        acc_t term1 = (acc_t)f_g * (acc_t)c_idx[i];
        acc_t term2 = (acc_t)i_g * (acc_t)g_g;
        acc_t next_c = term1 + term2;

        c_odx[i] = (data_t)next_c;
        data_t tanh_c = tanh_fast(next_c);
        h_odx[i] = (data_t)((acc_t)o_g * (acc_t)tanh_c);
    }
}

// compute core
static void compute(hls::stream<data_t>& fifo_in, hls::stream<data_t>& fifo_out, bool restart) {
    static data_t h0[128], c0[128], h1[128], c1[128];
#pragma HLS ARRAY_PARTITION variable=h0 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=c0 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=h1 cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=c1 cyclic factor=4

    if (restart) {
        RESET_LOOP: for (int i = 0; i < 128; i++) {
#pragma HLS UNROLL
            h0[i] = 0; c0[i] = 0; h1[i] = 0; c1[i] = 0;
        }
    }

    data_t in_buf[257];
    data_t fc1_out[128];
    data_t fc2_out[257];

    READ_FRAME: for (int i = 0; i < 257; i++) {
#pragma HLS PIPELINE II=1
        in_buf[i] = fifo_in.read();
    }

    lstm_cell_opt<257, 385>(in_buf, h0, c0, h0, c0, W_l0_fixed, B_l0_fixed);
    lstm_cell_opt<128, 256>(h0, h1, c1, h1, c1, W_l1_fixed, B_l1_fixed);

    FC1_LOOP: for (int i = 0; i < 128; i++) {
#pragma HLS PIPELINE II=1
        acc_t_fc acc = (acc_t_fc)b_fc1_fixed[i];
        for (int j = 0; j < 128; j++) {
            acc += (acc_t_fc)h1[j] * (acc_t_fc)W_fc1_fixed[i][j];
        }
        data_t res = (data_t)acc;
        fc1_out[i] = (res > (data_t)0) ? res : (data_t)(res * (data_t)0.3);
    }

    FC2_LOOP: for (int i = 0; i < 257; i++) {
#pragma HLS PIPELINE II=1
        acc_t_fc acc = (acc_t_fc)b_fc2_fixed[i];
        for (int j = 0; j < 128; j++) {
            acc += (acc_t_fc)fc1_out[j] * (acc_t_fc)W_fc2_fixed[i][j];
        }
        fc2_out[i] = (data_t)acc;
    }

    WRITE_FIFO: for (int i = 0; i < 257; i++) {
#pragma HLS PIPELINE II=1
        fifo_out.write(fc2_out[i]);
    }
}

void read_input(hls::stream<axis_t> &in_stream, hls::stream<data_t> &fifo_in) {
    for (int i = 0; i < 257; i++) {
        #pragma HLS PIPELINE II=1
        axis_t pkt = in_stream.read();

        data_t x;

        x.range() = pkt.data.range(15, 0);

        fifo_in.write(x);
    }
}

void write_output(hls::stream<data_t>& fifo_out, hls::stream<axis_t>& out_stream) {
    const data_t MASK_MIN = (data_t)0.05;

    for (int i = 0; i < 257; i++) {
#pragma HLS PIPELINE II=1
        data_t x = fifo_out.read();
        acc_t x_slope = (acc_t)x * (acc_t)slope_fixed[i];
        data_t sig = sigmoid_fast(x_slope);
        data_t mask = (data_t)1.2 * sig;

        if (mask < MASK_MIN) mask = MASK_MIN;

        axis_t pkt;
        pkt.data = 0;
        pkt.data.range(15, 0) = mask.range();
        pkt.last = (i == 256);
        pkt.keep = -1;
        out_stream.write(pkt);
    }
}

// top
void generator_top(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream, bool restart) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=restart bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

#pragma HLS BIND_STORAGE variable=W_l0_fixed type=rom_2p impl=bram
#pragma HLS BIND_STORAGE variable=W_l1_fixed type=rom_2p impl=bram

#pragma HLS BIND_STORAGE variable=W_fc1_fixed type=rom_1p impl=lutram
#pragma HLS BIND_STORAGE variable=W_fc2_fixed type=rom_1p impl=lutram
#pragma HLS BIND_STORAGE variable=slope_fixed type=rom_1p impl=lutram
#pragma HLS BIND_STORAGE variable=b_fc1_fixed type=rom_1p impl=lutram
#pragma HLS BIND_STORAGE variable=b_fc2_fixed type=rom_1p impl=lutram

#pragma HLS DATAFLOW
    hls::stream<data_t> fifo_in("in");
    hls::stream<data_t> fifo_out("out");
#pragma HLS STREAM variable=fifo_in depth=512
#pragma HLS STREAM variable=fifo_out depth=512

    read_input(in_stream, fifo_in);
    compute(fifo_in, fifo_out, restart);
    write_output(fifo_out, out_stream);
}
