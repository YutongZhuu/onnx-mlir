#ifndef MYACCEL_CONV3X3_KERNEL_H
#define MYACCEL_CONV3X3_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

// The current YOLO model tops out at 128 input channels for a 3x3
// convolution. The host runtime rejects larger layers before launching HLS.
#define MYACCEL_CONV3X3_MAX_INPUT_CHANNELS 128
#define MYACCEL_CONV3X3_MAX_STRIDE 2

void conv3x3_kernel(const float *x, const float *weight, const float *bias,
    float *y, int n_size, int c_size, int h_size, int input_w_size,
    int m_size, int oh_size, int ow_size, int pad_left, int pad_top,
    int stride_h, int stride_w, int has_bias);

#ifdef __cplusplus
}
#endif

#endif
