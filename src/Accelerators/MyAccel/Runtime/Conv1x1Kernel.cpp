#include "Conv1x1Kernel.h"

#include <stdint.h>

namespace {

// Four input channels and four output channels are evaluated in parallel.
// Sixteen output channels share each input tile so that NCHW input data is not
// reread once per output channel.
constexpr int kInputParallel = 4;
constexpr int kOutputParallel = 4;
constexpr int kOutputBlock = 16;
constexpr int kPixelTile = 16;

static float dot4(float lhs0, float lhs1, float lhs2, float lhs3, float rhs0,
    float rhs1, float rhs2, float rhs3) {
#pragma HLS INLINE
  const float product0 = lhs0 * rhs0;
  const float product1 = lhs1 * rhs1;
  const float product2 = lhs2 * rhs2;
  const float product3 = lhs3 * rhs3;
  return (product0 + product1) + (product2 + product3);
}

} // namespace

// Computes a 1x1, stride-1, pad-0, dilation-1, group-1 convolution.
//
// Logically this is GEMM:
//   [N*H*W, Cin] x [Cin, Cout] -> [N*H*W, Cout]
//
// The external tensors remain NCHW/OIHW. Small on-chip tiles perform the
// logical NHW flattening without materializing a whole-tensor NCHW-to-NHWC
// transpose in DDR.
extern "C" void conv1x1_kernel(const float *x, const float *weight,
    const float *bias, float *y, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int has_bias) {
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = weight offset = slave bundle = gmem1 \
    max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = bias offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = y offset = slave bundle = gmem3 \
    max_write_burst_length = 64 num_write_outstanding = 16
#pragma HLS INTERFACE s_axilite port = x bundle = control
#pragma HLS INTERFACE s_axilite port = weight bundle = control
#pragma HLS INTERFACE s_axilite port = bias bundle = control
#pragma HLS INTERFACE s_axilite port = y bundle = control
#pragma HLS INTERFACE s_axilite port = n_size bundle = control
#pragma HLS INTERFACE s_axilite port = c_size bundle = control
#pragma HLS INTERFACE s_axilite port = h_size bundle = control
#pragma HLS INTERFACE s_axilite port = input_w_size bundle = control
#pragma HLS INTERFACE s_axilite port = m_size bundle = control
#pragma HLS INTERFACE s_axilite port = has_bias bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  if (n_size <= 0 || c_size <= 0 ||
      c_size > MYACCEL_CONV1X1_MAX_INPUT_CHANNELS || h_size <= 0 ||
      input_w_size <= 0 || m_size <= 0)
    return;

  const int spatial_size = h_size * input_w_size;

OutputBlockLoop:
  for (int m_block = 0; m_block < m_size; m_block += kOutputBlock) {
    float weight_cache[kOutputBlock][MYACCEL_CONV1X1_MAX_INPUT_CHANNELS];
    float bias_cache[kOutputBlock];
#pragma HLS ARRAY_PARTITION variable = weight_cache cyclic \
    factor = kOutputParallel dim = 1
#pragma HLS ARRAY_PARTITION variable = weight_cache cyclic \
    factor = kInputParallel dim = 2
#pragma HLS ARRAY_PARTITION variable = bias_cache complete
#pragma HLS BIND_STORAGE variable = weight_cache type = ram_2p impl = bram

  LoadBiasLoop:
    for (int local_m = 0; local_m < kOutputBlock; ++local_m) {
#pragma HLS PIPELINE II = 1
      const int m = m_block + local_m;
      bias_cache[local_m] =
          (has_bias && m < m_size) ? bias[m] : 0.0f;
    }

  LoadWeightOutputLoop:
    for (int local_m = 0; local_m < kOutputBlock; ++local_m) {
      const int m = m_block + local_m;
    LoadWeightChannelLoop:
      for (int c = 0; c < c_size; ++c) {
#pragma HLS PIPELINE II = 1
        weight_cache[local_m][c] =
            m < m_size ? weight[(uint32_t)m * c_size + c] : 0.0f;
      }
    }

  BatchLoop:
    for (int n = 0; n < n_size; ++n) {
    PixelTileLoop:
      for (int pixel_base = 0; pixel_base < spatial_size;
           pixel_base += kPixelTile) {
        float input_tile[kInputParallel][kPixelTile];
        float accum[kOutputBlock][kPixelTile];
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = accum complete dim = 1
#pragma HLS ARRAY_PARTITION variable = accum complete dim = 2

      InitAccumPixelLoop:
        for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
        InitAccumOutputLoop:
          for (int local_m = 0; local_m < kOutputBlock; ++local_m) {
#pragma HLS UNROLL
            accum[local_m][pixel] = bias_cache[local_m];
          }
        }

      InputChannelTileLoop:
        for (int c_base = 0; c_base < c_size;
             c_base += kInputParallel) {
        LoadInputChannelLoop:
          for (int input_lane = 0; input_lane < kInputParallel;
               ++input_lane) {
            const int c = c_base + input_lane;
          LoadInputPixelLoop:
            for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
              const int spatial = pixel_base + pixel;
              if (c < c_size && spatial < spatial_size) {
                const uint32_t x_index =
                    ((uint32_t)n * c_size + c) * spatial_size + spatial;
                input_tile[input_lane][pixel] = x[x_index];
              } else {
                input_tile[input_lane][pixel] = 0.0f;
              }
            }
          }

        OutputSubBlockLoop:
          for (int output_base = 0; output_base < kOutputBlock;
               output_base += kOutputParallel) {
          ComputePixelLoop:
            for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = accum inter false
            ComputeOutputLaneLoop:
              for (int output_lane = 0; output_lane < kOutputParallel;
                   ++output_lane) {
#pragma HLS UNROLL
                const int local_m = output_base + output_lane;
                const float weight0 =
                    c_base < c_size ? weight_cache[local_m][c_base] : 0.0f;
                const float weight1 = c_base + 1 < c_size
                                          ? weight_cache[local_m][c_base + 1]
                                          : 0.0f;
                const float weight2 = c_base + 2 < c_size
                                          ? weight_cache[local_m][c_base + 2]
                                          : 0.0f;
                const float weight3 = c_base + 3 < c_size
                                          ? weight_cache[local_m][c_base + 3]
                                          : 0.0f;
                accum[local_m][pixel] +=
                    dot4(input_tile[0][pixel], input_tile[1][pixel],
                        input_tile[2][pixel], input_tile[3][pixel], weight0,
                        weight1, weight2, weight3);
              }
            }
          }
        }

      StoreOutputLoop:
        for (int local_m = 0; local_m < kOutputBlock; ++local_m) {
          const int m = m_block + local_m;
        StoreOutputPixelLoop:
          for (int pixel = 0; pixel < kPixelTile; ++pixel) {
#pragma HLS PIPELINE II = 1
            const int spatial = pixel_base + pixel;
            if (m < m_size && spatial < spatial_size) {
              const uint32_t y_index =
                  ((uint32_t)n * m_size + m) * spatial_size + spatial;
              y[y_index] = accum[local_m][pixel];
            }
          }
        }
      }
    }
  }
}
