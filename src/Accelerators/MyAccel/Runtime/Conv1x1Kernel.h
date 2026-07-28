#ifndef MYACCEL_CONV1X1_KERNEL_H
#define MYACCEL_CONV1X1_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

// The current YOLO model tops out at 512 input channels for a 1x1
// convolution. The host runtime rejects larger layers before launching HLS.
#define MYACCEL_CONV1X1_MAX_INPUT_CHANNELS 512

void conv1x1_kernel(const float *x, const float *weight, const float *bias,
    float *y, int n_size, int c_size, int h_size, int input_w_size,
    int m_size, int has_bias);

#ifdef __cplusplus
}
#endif

#endif
