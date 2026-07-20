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

#ifdef __cplusplus
}
#endif

#endif
