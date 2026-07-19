#ifndef MYACCEL_CONV2D_KERNEL_H
#define MYACCEL_CONV2D_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void conv2d_kernel(const float *x, const float *weight, const float *bias,
    float *y, int64_t n_size, int64_t c_size, int64_t h_size,
    int64_t input_w_size, int64_t m_size, int64_t kh_size, int64_t kw_size,
    int64_t oh_size, int64_t ow_size, int64_t dilation_h, int64_t dilation_w,
    int64_t c_per_group, int64_t group, int64_t pad_left, int64_t pad_top,
    int64_t stride_h, int64_t stride_w, int64_t has_bias);

#ifdef __cplusplus
}
#endif

#endif
