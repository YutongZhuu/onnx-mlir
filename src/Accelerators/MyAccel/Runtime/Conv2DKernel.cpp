#include "Conv2DKernel.h"

// This version of kernel is correctness based, no buffer/reuse logic at all
void conv2d_kernel(
  const float *x, // In NCHW, N = batch size, C = Channel, H = Height, W = width
  const float *weight, // In OIHW, O = Num of filters, I = Channels per group, HW = kernel height /width
  const float *bias, // One scalar per output channel.
  float *y, // Also in NCHW
  // Metadata
  int64_t n_size, 
  int64_t c_size, 
  int64_t h_size,
  int64_t input_w_size, 
  int64_t m_size, 
  int64_t kh_size,
  int64_t kw_size, 
  int64_t oh_size, 
  int64_t ow_size,

  /*
  Normal 3x3 kernel:
  x x x
  x x x
  x x x

  Kernel with both h and w dilation:
  x . x . x
  . . . . .
  x . x . x
  . . . . .
  x . x . x
  */
  int64_t dilation_h, 
  int64_t dilation_w, 

  int64_t c_per_group, // How many channel are in one group
  int64_t group, 

  int64_t pad_left,
  int64_t pad_top, 

  int64_t stride_h, 
  int64_t stride_w,

  int64_t has_bias // If bias exist = 1, if not = 0
) {
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = weight offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = bias offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = y offset = slave bundle = gmem3
#pragma HLS INTERFACE s_axilite port = x bundle = control
#pragma HLS INTERFACE s_axilite port = weight bundle = control
#pragma HLS INTERFACE s_axilite port = bias bundle = control
#pragma HLS INTERFACE s_axilite port = y bundle = control
#pragma HLS INTERFACE s_axilite port = n_size bundle = control
#pragma HLS INTERFACE s_axilite port = c_size bundle = control
#pragma HLS INTERFACE s_axilite port = h_size bundle = control
#pragma HLS INTERFACE s_axilite port = input_w_size bundle = control
#pragma HLS INTERFACE s_axilite port = m_size bundle = control
#pragma HLS INTERFACE s_axilite port = c_per_group bundle = control
#pragma HLS INTERFACE s_axilite port = kh_size bundle = control
#pragma HLS INTERFACE s_axilite port = kw_size bundle = control
#pragma HLS INTERFACE s_axilite port = oh_size bundle = control
#pragma HLS INTERFACE s_axilite port = ow_size bundle = control
#pragma HLS INTERFACE s_axilite port = dilation_h bundle = control
#pragma HLS INTERFACE s_axilite port = dilation_w bundle = control
#pragma HLS INTERFACE s_axilite port = group bundle = control
#pragma HLS INTERFACE s_axilite port = pad_left bundle = control
#pragma HLS INTERFACE s_axilite port = pad_top bundle = control
#pragma HLS INTERFACE s_axilite port = stride_h bundle = control
#pragma HLS INTERFACE s_axilite port = stride_w bundle = control
#pragma HLS INTERFACE s_axilite port = has_bias bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  const int64_t m_per_group = m_size / group;

  for (int64_t n = 0; n < n_size; ++n) {
    for (int64_t m = 0; m < m_size; ++m) {
      const int64_t g = m / m_per_group;
      for (int64_t oh = 0; oh < oh_size; ++oh) {
        for (int64_t ow = 0; ow < ow_size; ++ow) {
#pragma HLS PIPELINE II = 1
          float sum = has_bias ? bias[m] : 0.0f;
          for (int64_t cg = 0; cg < c_per_group; ++cg) {
            for (int64_t kh = 0; kh < kh_size; ++kh) {
              for (int64_t kw = 0; kw < kw_size; ++kw) {
                const int64_t ih = oh * stride_h + kh * dilation_h - pad_top;
                const int64_t iw = ow * stride_w + kw * dilation_w - pad_left;
                if (ih < 0 || ih >= h_size || iw < 0 ||
                    iw >= input_w_size)
                  continue;

                const int64_t c = g * c_per_group + cg;
                const int64_t x_index =
                    ((n * c_size + c) * h_size + ih) * input_w_size + iw;
                const int64_t w_index =
                    ((m * c_per_group + cg) * kh_size + kh) * kw_size + kw;
                sum += x[x_index] * weight[w_index];
              }
            }
          }

          const int64_t y_index =
              ((n * m_size + m) * oh_size + oh) * ow_size + ow;
          y[y_index] = sum;
        }
      }
    }
  }
}
