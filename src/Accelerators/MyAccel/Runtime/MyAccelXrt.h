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

// Dispatches a supported 1x1, 3x3, or specialized 6x6-stem signed-int8
// convolution, including bias, the bit-exact float32 requantization contract,
// and the final INT8 clamp. The 6x6 path also requires a matching xclbin and
// MYACCEL_ENABLE_6X6_STEM=1.
int myaccel_xrt_conv2d_i8(const int8_t *x, const int8_t *weight,
    const int32_t *bias, int8_t *y, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int kh_size, int kw_size, int oh_size,
    int ow_size, int dilation_h, int dilation_w, int c_per_group, int group,
    int pad_left, int pad_top, int stride_h, int stride_w, int x_zero_point,
    int w_zero_point, uint32_t requant_multiplier_bits,
    int output_zero_point);

#ifdef __cplusplus
}
#endif

#endif
