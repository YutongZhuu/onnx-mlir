#include <math.h>
#include <stdio.h>

#include "Conv2DKernel.h"

int main() {
  float input[16];
  for (int i = 0; i < 16; ++i)
    input[i] = (float)(i + 1);

  float weight[9];
  for (int i = 0; i < 9; ++i)
    weight[i] = 1.0f;

  float bias[1] = {0.5f};
  float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float expected[4] = {54.5f, 63.5f, 90.5f, 99.5f};

  conv2d_kernel(input, weight, bias, output,
      /*n_size=*/1,
      /*c_size=*/1,
      /*h_size=*/4,
      /*input_w_size=*/4,
      /*m_size=*/1,
      /*kh_size=*/3,
      /*kw_size=*/3,
      /*oh_size=*/2,
      /*ow_size=*/2,
      /*dilation_h=*/1,
      /*dilation_w=*/1,
      /*c_per_group=*/1,
      /*group=*/1,
      /*pad_left=*/0,
      /*pad_top=*/0,
      /*stride_h=*/1,
      /*stride_w=*/1,
      /*has_bias=*/1);

  int ok = 1;
  for (int i = 0; i < 4; ++i) {
    printf("y[%d] = %.1f (expected %.1f)\n", i, output[i], expected[i]);
    if (fabsf(output[i] - expected[i]) > 1e-5f)
      ok = 0;
  }

  puts(ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
