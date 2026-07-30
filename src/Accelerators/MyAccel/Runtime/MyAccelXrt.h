#ifndef MYACCEL_XRT_H
#define MYACCEL_XRT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int myaccel_xrt_conv2d_f32(const float *x, const float *weight, const float *bias,
    float *y, int n_size, int c_size, int h_size, int input_w_size,
    int m_size, int kh_size, int kw_size, int oh_size, int ow_size,
    int dilation_h, int dilation_w, int c_per_group, int group, int pad_left,
    int pad_top, int stride_h, int stride_w, int has_bias);

// Dispatches supported 1x1 or 3x3 signed-int8 convolution dot products and
// returns raw int32 accumulators. Bias, scales, requantization, and Q/DQ stay
// on the host.
int myaccel_xrt_conv2d_i8(const int8_t *x, const int8_t *weight,
    int32_t *accumulator, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int kh_size, int kw_size, int oh_size,
    int ow_size, int dilation_h, int dilation_w, int c_per_group, int group,
    int pad_left, int pad_top, int stride_h, int stride_w,
    int x_zero_point, int w_zero_point);

#ifdef __cplusplus
}
#endif

#endif
