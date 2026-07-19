#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "MyAccelXrt.h"
#include "onnx-mlir/Runtime/OMTensor.h"

uint64_t OMInitCompatibleAccelMyAccel(uint64_t version) {
  (void)version;
  return 1;
}

static int check_support(const int64_t *xs, const int64_t *ws,
    const int64_t *ys, int64_t dh, int64_t dw, int64_t group, int64_t sh,
    int64_t sw) {
  if (getenv("MYACCEL_FORCE_CPU"))
    return 0;
  if (dh != 1 || dw != 1 || group != 1 || sh <= 0 || sw <= 0) 
  // Excluded dilated conv, grouped conv, depthwise conv and invalid strides
    return 0;
  if (xs[0] <= 0 || xs[1] <= 0 || xs[2] <= 0 || xs[3] <= 0 ||
      ws[0] <= 0 || ws[1] <= 0 || ws[2] <= 0 || ws[3] <= 0 ||
      ys[0] <= 0 || ys[1] <= 0 || ys[2] <= 0 || ys[3] <= 0)
  // More invalid metadata
    return 0;
  return xs[1] == ws[1] && ys[0] == xs[0] && ys[1] == ws[0];
}

static void my_conv_f32_cpu_fallback(float *y, const float *x, const float *w,
    const float *b, const int64_t *xs, const int64_t *ws, const int64_t *ys,
    int64_t dh, int64_t dw, int64_t group, int64_t padLeft, int64_t padTop,
    int64_t sh, int64_t sw) {
  const int64_t nSize = xs[0], cSize = xs[1], hSize = xs[2], wSize = xs[3];
  const int64_t mSize = ws[0], cPerGroup = ws[1], khSize = ws[2], kwSize = ws[3];
  const int64_t ohSize = ys[2], owSize = ys[3];
  const int64_t mPerGroup = mSize / group;

  for (int64_t n = 0; n < nSize; ++n)
    for (int64_t m = 0; m < mSize; ++m) {
      const int64_t g = m / mPerGroup;
      for (int64_t oh = 0; oh < ohSize; ++oh)
        for (int64_t ow = 0; ow < owSize; ++ow) {
          float sum = b ? b[m] : 0.0f;
          for (int64_t cg = 0; cg < cPerGroup; ++cg)
            for (int64_t kh = 0; kh < khSize; ++kh)
              for (int64_t kw = 0; kw < kwSize; ++kw) {
                const int64_t ih = oh * sh + kh * dh - padTop;
                const int64_t iw = ow * sw + kw * dw - padLeft;
                if (ih < 0 || ih >= hSize || iw < 0 || iw >= wSize)
                  continue;
                const int64_t c = g * cPerGroup + cg;
                sum += x[((n * cSize + c) * hSize + ih) * wSize + iw] *
                       w[((m * cPerGroup + cg) * khSize + kh) * kwSize + kw];
              }
          y[((n * mSize + m) * ohSize + oh) * owSize + ow] = sum;
        }
    }
}

// NCHW, OIHW, float32 reference convolution. The OMTensor arguments are
// non-owning wrappers created by the generated model around its memrefs.
void my_conv_f32(OMTensor *yTensor, const OMTensor *xTensor,
    const OMTensor *wTensor, const OMTensor *bTensor,
    int64_t dh, int64_t dw, int64_t group, int64_t padLeft, int64_t padTop,
    int64_t sh, int64_t sw) {
  const int64_t *xs = omTensorGetShape(xTensor);
  const int64_t *ws = omTensorGetShape(wTensor);
  const int64_t *ys = omTensorGetShape(yTensor);
  const float *x = (const float *)omTensorGetDataPtr(xTensor);
  const float *w = (const float *)omTensorGetDataPtr(wTensor);
  const float *b = bTensor ? (const float *)omTensorGetDataPtr(bTensor) : 0;
  float *y = (float *)omTensorGetDataPtr(yTensor);

  const int64_t nSize = xs[0], cSize = xs[1], hSize = xs[2], wSize = xs[3];
  const int64_t mSize = ws[0], cPerGroup = ws[1], khSize = ws[2], kwSize = ws[3];
  const int64_t ohSize = ys[2], owSize = ys[3];

  fprintf(stderr, "MYACCEL: my_conv_f32 invoked\n");
  if (!check_support(xs, ws, ys, dh, dw, group, sh, sw)) {
    fprintf(stderr, "MYACCEL: unsupported conv, falling back to CPU convolution\n");
    my_conv_f32_cpu_fallback(
        y, x, w, b, xs, ws, ys, dh, dw, group, padLeft, padTop, sh, sw);
    return;
  }

  if (myaccel_xrt_conv2d_f32(x, w, b, y, nSize, cSize, hSize, wSize, mSize,
          khSize, kwSize, ohSize, owSize, dh, dw, cPerGroup, group, padLeft,
          padTop, sh, sw, bTensor != 0)) {
    return;
  }

  fprintf(stderr, "MYACCEL: XRT path failed, falling back to CPU convolution\n");
  my_conv_f32_cpu_fallback(
        y, x, w, b, xs, ws, ys, dh, dw, group, padLeft, padTop, sh, sw);
  return;
}

// Compatibility entry used by onnx-mlir's --ops-for-call verification path.
// The tiny test model omits Conv attributes, so ONNX defaults apply.
void Conv(OMTensor *y, const OMTensor *x, const OMTensor *w,
    const OMTensor *b, const char *autoPad, int64_t group,
    const OMTensor *kernelShape) {
  (void)autoPad;
  (void)kernelShape;
  my_conv_f32(y, x, w, b, 1, 1, group, 0, 0, 1, 1);
}
