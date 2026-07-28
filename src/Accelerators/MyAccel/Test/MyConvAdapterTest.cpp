#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "onnx-mlir/Runtime/OMTensor.h"

extern "C" void my_conv_f32(OMTensor *yTensor, const OMTensor *xTensor,
    const OMTensor *wTensor, const OMTensor *bTensor, int64_t dh, int64_t dw,
    int64_t group, int64_t padLeft, int64_t padTop, int64_t sh, int64_t sw);

int main() {
  float input[16];
  for (int i = 0; i < 16; ++i)
    input[i] = (float)(i + 1);

  float weight[9];
  for (int i = 0; i < 9; ++i)
    weight[i] = 1.0f;

  float bias[1] = {0.5f};
  float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  int64_t inputShape[4] = {1, 1, 4, 4};
  int64_t weightShape[4] = {1, 1, 3, 3};
  int64_t biasShape[1] = {1};
  int64_t outputShape[4] = {1, 1, 2, 2};

  OMTensor *xTensor = omTensorCreate(input, inputShape, 4, ONNX_TYPE_FLOAT);
  OMTensor *wTensor = omTensorCreate(weight, weightShape, 4, ONNX_TYPE_FLOAT);
  OMTensor *bTensor = omTensorCreate(bias, biasShape, 1, ONNX_TYPE_FLOAT);
  OMTensor *yTensor = omTensorCreate(output, outputShape, 4, ONNX_TYPE_FLOAT);

  if (!xTensor || !wTensor || !bTensor || !yTensor) {
    puts("FAIL: could not create OMTensor inputs");
    return 1;
  }

  my_conv_f32(yTensor, xTensor, wTensor, bTensor,
      /*dh=*/1,
      /*dw=*/1,
      /*group=*/1,
      /*padLeft=*/0,
      /*padTop=*/0,
      /*sh=*/1,
      /*sw=*/1);

  const float expected[4] = {54.5f, 63.5f, 90.5f, 99.5f};
  int ok = 1;
  for (int i = 0; i < 4; ++i) {
    printf("y[%d] = %.1f (expected %.1f)\n", i, output[i], expected[i]);
    if (fabsf(output[i] - expected[i]) > 1e-5f)
      ok = 0;
  }

  omTensorDestroy(xTensor);
  omTensorDestroy(wTensor);
  omTensorDestroy(bTensor);
  omTensorDestroy(yTensor);

  puts(ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
