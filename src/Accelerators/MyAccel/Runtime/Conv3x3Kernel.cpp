#include "Conv3x3Kernel.h"

#include <stdint.h>

namespace {

// Each compute cycle contains 4 output channels x 2 input channels x 9
// spatial taps = 72 floating-point multipliers. The nine 3x3 taps are fully
// unrolled and reduced through a balanced tree.
constexpr int kInputParallel = 2;
constexpr int kOutputParallel = 4;
constexpr int kOutputBlock = 16;
constexpr int kOutputTileHeight = 2;
constexpr int kOutputTileWidth = 8;
constexpr int kInputTileHeight =
    (kOutputTileHeight - 1) * MYACCEL_CONV3X3_MAX_STRIDE + 3;
constexpr int kInputTileWidth =
    (kOutputTileWidth - 1) * MYACCEL_CONV3X3_MAX_STRIDE + 3;

static float reduce18(const float value[18]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = value complete
  const float level1_0 = value[0] + value[1];
  const float level1_1 = value[2] + value[3];
  const float level1_2 = value[4] + value[5];
  const float level1_3 = value[6] + value[7];
  const float level1_4 = value[8] + value[9];
  const float level1_5 = value[10] + value[11];
  const float level1_6 = value[12] + value[13];
  const float level1_7 = value[14] + value[15];
  const float level1_8 = value[16] + value[17];

  const float level2_0 = level1_0 + level1_1;
  const float level2_1 = level1_2 + level1_3;
  const float level2_2 = level1_4 + level1_5;
  const float level2_3 = level1_6 + level1_7;

  const float level3_0 = level2_0 + level2_1;
  const float level3_1 = level2_2 + level2_3;
  return (level3_0 + level3_1) + level1_8;
}

} // namespace

// Computes a 3x3, dilation-1, group-1 convolution in NCHW/OIHW layout.
// Stride 1 and 2 are supported. A small overlapping input patch is cached on
// chip and shared by a block of 16 output channels.
extern "C" void conv3x3_kernel(const float *x, const float *weight,
    const float *bias, float *y, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int oh_size, int ow_size, int pad_left,
    int pad_top, int stride_h, int stride_w, int has_bias) {
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
#pragma HLS INTERFACE s_axilite port = oh_size bundle = control
#pragma HLS INTERFACE s_axilite port = ow_size bundle = control
#pragma HLS INTERFACE s_axilite port = pad_left bundle = control
#pragma HLS INTERFACE s_axilite port = pad_top bundle = control
#pragma HLS INTERFACE s_axilite port = stride_h bundle = control
#pragma HLS INTERFACE s_axilite port = stride_w bundle = control
#pragma HLS INTERFACE s_axilite port = has_bias bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  if (n_size <= 0 || c_size <= 0 ||
      c_size > MYACCEL_CONV3X3_MAX_INPUT_CHANNELS || h_size <= 0 ||
      input_w_size <= 0 || m_size <= 0 || oh_size <= 0 || ow_size <= 0 ||
      stride_h <= 0 || stride_h > MYACCEL_CONV3X3_MAX_STRIDE ||
      stride_w <= 0 || stride_w > MYACCEL_CONV3X3_MAX_STRIDE)
    return;

OutputBlockLoop:
  for (int m_block = 0; m_block < m_size; m_block += kOutputBlock) {
    float weight_cache[kOutputBlock][MYACCEL_CONV3X3_MAX_INPUT_CHANNELS][3][3];
    float bias_cache[kOutputBlock];
#pragma HLS ARRAY_PARTITION variable = weight_cache cyclic \
    factor = kOutputParallel dim = 1
#pragma HLS ARRAY_PARTITION variable = weight_cache cyclic \
    factor = kInputParallel dim = 2
#pragma HLS ARRAY_PARTITION variable = weight_cache complete dim = 3
#pragma HLS ARRAY_PARTITION variable = weight_cache complete dim = 4
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
      LoadWeightRowLoop:
        for (int kh = 0; kh < 3; ++kh) {
        LoadWeightColumnLoop:
          for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II = 1
            const uint32_t w_index =
                (((uint32_t)m * c_size + c) * 3 + kh) * 3 + kw;
            weight_cache[local_m][c][kh][kw] =
                m < m_size ? weight[w_index] : 0.0f;
          }
        }
      }
    }

  BatchLoop:
    for (int n = 0; n < n_size; ++n) {
    OutputTileRowLoop:
      for (int oh_base = 0; oh_base < oh_size;
           oh_base += kOutputTileHeight) {
      OutputTileColumnLoop:
        for (int ow_base = 0; ow_base < ow_size;
             ow_base += kOutputTileWidth) {
          float input_tile[kInputParallel][kInputTileHeight][kInputTileWidth];
          float accum[kOutputBlock][kOutputTileHeight][kOutputTileWidth];
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 2
#pragma HLS ARRAY_PARTITION variable = input_tile complete dim = 3
#pragma HLS ARRAY_PARTITION variable = accum cyclic \
    factor = kOutputParallel dim = 1
#pragma HLS ARRAY_PARTITION variable = accum complete dim = 2
#pragma HLS ARRAY_PARTITION variable = accum complete dim = 3

        InitAccumOutputLoop:
          for (int local_m = 0; local_m < kOutputBlock; ++local_m) {
          InitAccumRowLoop:
            for (int local_oh = 0; local_oh < kOutputTileHeight; ++local_oh) {
            InitAccumColumnLoop:
              for (int local_ow = 0; local_ow < kOutputTileWidth; ++local_ow) {
#pragma HLS PIPELINE II = 1
                accum[local_m][local_oh][local_ow] = bias_cache[local_m];
              }
            }
          }

        InputChannelTileLoop:
          for (int c_base = 0; c_base < c_size;
               c_base += kInputParallel) {
            const int remaining_oh = oh_size - oh_base;
            const int remaining_ow = ow_size - ow_base;
            const int valid_oh = remaining_oh < kOutputTileHeight
                                     ? remaining_oh
                                     : kOutputTileHeight;
            const int valid_ow = remaining_ow < kOutputTileWidth
                                     ? remaining_ow
                                     : kOutputTileWidth;
            const int patch_height = (valid_oh - 1) * stride_h + 3;
            const int patch_width = (valid_ow - 1) * stride_w + 3;
            const int input_row_base = oh_base * stride_h - pad_top;
            const int input_column_base = ow_base * stride_w - pad_left;

          LoadInputChannelLoop:
            for (int input_lane = 0; input_lane < kInputParallel;
                 ++input_lane) {
              const int c = c_base + input_lane;
            LoadInputRowLoop:
              for (int local_ih = 0; local_ih < kInputTileHeight; ++local_ih) {
              LoadInputColumnLoop:
                for (int local_iw = 0; local_iw < kInputTileWidth;
                     ++local_iw) {
#pragma HLS PIPELINE II = 1
                  const int ih = input_row_base + local_ih;
                  const int iw = input_column_base + local_iw;
                  if (c < c_size && local_ih < patch_height &&
                      local_iw < patch_width && ih >= 0 && ih < h_size &&
                      iw >= 0 && iw < input_w_size) {
                    const uint32_t x_index =
                        ((uint32_t)n * c_size + c) * h_size * input_w_size +
                        (uint32_t)ih * input_w_size + iw;
                    input_tile[input_lane][local_ih][local_iw] = x[x_index];
                  } else {
                    input_tile[input_lane][local_ih][local_iw] = 0.0f;
                  }
                }
              }
            }

          OutputSubBlockLoop:
            for (int output_base = 0; output_base < kOutputBlock;
                 output_base += kOutputParallel) {
            ComputeOutputRowLoop:
              for (int local_oh = 0; local_oh < kOutputTileHeight;
                   ++local_oh) {
              ComputeOutputColumnLoop:
                for (int local_ow = 0; local_ow < kOutputTileWidth;
                     ++local_ow) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = accum inter false
                ComputeOutputLaneLoop:
                  for (int output_lane = 0;
                       output_lane < kOutputParallel; ++output_lane) {
#pragma HLS UNROLL
                    const int local_m = output_base + output_lane;
                    float products[18];
#pragma HLS ARRAY_PARTITION variable = products complete
                  ProductInputLaneLoop:
                    for (int input_lane = 0;
                         input_lane < kInputParallel; ++input_lane) {
#pragma HLS UNROLL
                      const int c = c_base + input_lane;
                    ProductRowLoop:
                      for (int kh = 0; kh < 3; ++kh) {
#pragma HLS UNROLL
                      ProductColumnLoop:
                        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS UNROLL
                          const int product_index =
                              (input_lane * 3 + kh) * 3 + kw;
                          const float weight_value =
                              c < c_size
                                  ? weight_cache[local_m][c][kh][kw]
                                  : 0.0f;
                          const int local_ih = local_oh * stride_h + kh;
                          const int local_iw = local_ow * stride_w + kw;
                          products[product_index] =
                              input_tile[input_lane][local_ih][local_iw] *
                              weight_value;
                        }
                      }
                    }
                    accum[local_m][local_oh][local_ow] +=
                        reduce18(products);
                  }
                }
              }
            }
          }

        StoreOutputLoop:
          for (int local_m = 0; local_m < kOutputBlock; ++local_m) {
            const int m = m_block + local_m;
          StoreOutputRowLoop:
            for (int local_oh = 0; local_oh < kOutputTileHeight; ++local_oh) {
              const int oh = oh_base + local_oh;
            StoreOutputColumnLoop:
              for (int local_ow = 0; local_ow < kOutputTileWidth;
                   ++local_ow) {
#pragma HLS PIPELINE II = 1
                const int ow = ow_base + local_ow;
                if (m < m_size && oh < oh_size && ow < ow_size) {
                  const uint32_t y_index =
                      ((uint32_t)n * m_size + m) * oh_size * ow_size +
                      (uint32_t)oh * ow_size + ow;
                  y[y_index] = accum[local_m][local_oh][local_ow];
                }
              }
            }
          }
        }
      }
    }
  }
}
