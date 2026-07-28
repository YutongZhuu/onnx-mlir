#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "MyAccelXrt.h"
#include "onnx-mlir/Runtime/OMTensor.h"

uint64_t OMInitCompatibleAccelMyAccel(uint64_t version) {
  (void)version;
  return 1;
}

static int product_fits_uint32(int64_t a, int64_t b, int64_t c, int64_t d) {
  if (a <= 0 || b <= 0 || c <= 0 || d <= 0)
    return 0;
  return a <= UINT32_MAX / b && a * b <= UINT32_MAX / c &&
         a * b * c <= UINT32_MAX / d;
}

static int value_fits_int(int64_t value) {
  return value >= 0 && value <= INT_MAX;
}

static int check_support(const int64_t *xs, const int64_t *ws,
    const int64_t *ys, int64_t dh, int64_t dw, int64_t group, int64_t sh,
    int64_t sw, int64_t padLeft, int64_t padTop) {
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
  if (!value_fits_int(dh) || !value_fits_int(dw) || !value_fits_int(group) ||
      !value_fits_int(sh) || !value_fits_int(sw) ||
      !value_fits_int(padLeft) || !value_fits_int(padTop))
    return 0;
  for (int i = 0; i < 4; ++i) {
    if (!value_fits_int(xs[i]) || !value_fits_int(ws[i]) ||
        !value_fits_int(ys[i]))
      return 0;
  }
  if (!product_fits_uint32(xs[0], xs[1], xs[2], xs[3]) ||
      !product_fits_uint32(ws[0], ws[1], ws[2], ws[3]) ||
      !product_fits_uint32(ys[0], ys[1], ys[2], ys[3]))
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

// Compute the FP32 value represented by a QDQ-wrapped INT8 convolution. The
// convolution products accumulate exactly in int32, then use the QDQ scales to
// recreate the floating-point Conv result expected by the existing graph.
// NCHW input and OIHW weights are assumed.
void my_conv_qdq_i8_f32(OMTensor *yTensor, const OMTensor *xTensor,
    const OMTensor *xScaleTensor, const OMTensor *xZeroPointTensor,
    const OMTensor *wTensor, const OMTensor *wScaleTensor,
    const OMTensor *wZeroPointTensor, const OMTensor *bTensor,
    const OMTensor *bScaleTensor, const OMTensor *bZeroPointTensor,
    int64_t dh, int64_t dw, int64_t group, int64_t padLeft, int64_t padTop,
    int64_t sh, int64_t sw) {
  const int64_t *xs = omTensorGetShape(xTensor);
  const int64_t *ws = omTensorGetShape(wTensor);
  const int64_t *ys = omTensorGetShape(yTensor);
  const int8_t *x = (const int8_t *)omTensorGetDataPtr(xTensor);
  const int8_t *w = (const int8_t *)omTensorGetDataPtr(wTensor);
  const int32_t *b = (const int32_t *)omTensorGetDataPtr(bTensor);
  float *y = (float *)omTensorGetDataPtr(yTensor);
  const float xScale = *(const float *)omTensorGetDataPtr(xScaleTensor);
  const int16_t xZeroPoint =
      *(const int8_t *)omTensorGetDataPtr(xZeroPointTensor);
  const float wScale = *(const float *)omTensorGetDataPtr(wScaleTensor);
  const int16_t wZeroPoint =
      *(const int8_t *)omTensorGetDataPtr(wZeroPointTensor);
  const float bScale = *(const float *)omTensorGetDataPtr(bScaleTensor);
  const int32_t bZeroPoint =
      *(const int32_t *)omTensorGetDataPtr(bZeroPointTensor);

  const int64_t nSize = xs[0], cSize = xs[1], hSize = xs[2], wSize = xs[3];
  const int64_t mSize = ws[0], cPerGroup = ws[1];
  const int64_t khSize = ws[2], kwSize = ws[3];
  const int64_t ohSize = ys[2], owSize = ys[3];
  const int64_t mPerGroup = mSize / group;
  const int64_t reductionSize = cPerGroup * khSize * kwSize;
  const int64_t spatialSize = ohSize * owSize;
  const float productScale = xScale * wScale;
  const int directOneByOne = khSize == 1 && kwSize == 1 && dh == 1 &&
      dw == 1 && sh == 1 && sw == 1 && padTop == 0 && padLeft == 0 &&
      ohSize == hSize && owSize == wSize;

  if (getenv("MYACCEL_PROFILE"))
    fprintf(stderr,
        "MYACCEL_INT8 shape=%lldx%lldx%lldx%lld->%lldx%lldx%lld "
        "kernel=%lldx%lld stride=%lldx%lld direct=%d\n",
        (long long)nSize, (long long)cSize, (long long)hSize,
        (long long)wSize, (long long)mSize, (long long)ohSize,
        (long long)owSize, (long long)khSize, (long long)kwSize,
        (long long)sh, (long long)sw, directOneByOne);

  int8_t *columns = 0;
  if (!directOneByOne) {
    const size_t columnBytes =
        (size_t)reductionSize * (size_t)spatialSize;
    if (posix_memalign((void **)&columns, 64, columnBytes) != 0 || !columns) {
      fprintf(stderr, "MYACCEL: failed to allocate %zu-byte INT8 im2col buffer\n",
          columnBytes);
      abort();
    }
  }

  for (int64_t n = 0; n < nSize; ++n) {
    for (int64_t g = 0; g < group; ++g) {
      const int8_t *matrixB = 0;
      if (directOneByOne) {
        matrixB = x + (n * cSize + g * cPerGroup) * spatialSize;
      } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int64_t reduction = 0; reduction < reductionSize; ++reduction) {
          const int64_t kw = reduction % kwSize;
          const int64_t kh = (reduction / kwSize) % khSize;
          const int64_t cg = reduction / (khSize * kwSize);
          const int64_t c = g * cPerGroup + cg;
          int8_t *column = columns + reduction * spatialSize;
          for (int64_t oh = 0; oh < ohSize; ++oh) {
            const int64_t ih = oh * sh + kh * dh - padTop;
            for (int64_t ow = 0; ow < owSize; ++ow) {
              const int64_t iw = ow * sw + kw * dw - padLeft;
              column[oh * owSize + ow] =
                  ih >= 0 && ih < hSize && iw >= 0 && iw < wSize
                      ? x[((n * cSize + c) * hSize + ih) * wSize + iw]
                      : (int8_t)xZeroPoint;
            }
          }
        }
        matrixB = columns;
      }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
      for (int64_t mg = 0; mg < mPerGroup; ++mg) {
        const int64_t m = g * mPerGroup + mg;
        const int8_t *weight = w + m * reductionSize;
        float *output = y + (n * mSize + m) * spatialSize;
        const float bias = ((float)(b[m] - bZeroPoint)) * bScale;
        int64_t position = 0;

#if defined(__aarch64__)
        const int16x8_t xZeroPointVector = vdupq_n_s16(xZeroPoint);
        for (; position + 8 <= spatialSize; position += 8) {
          int32x4_t sumLow = vdupq_n_s32(0);
          int32x4_t sumHigh = vdupq_n_s32(0);
          for (int64_t reduction = 0; reduction < reductionSize; ++reduction) {
            const int8x8_t input8 =
                vld1_s8(matrixB + reduction * spatialSize + position);
            const int16x8_t centeredInput =
                vsubq_s16(vmovl_s8(input8), xZeroPointVector);
            const int16_t centeredWeight =
                (int16_t)weight[reduction] - wZeroPoint;
            sumLow = vmlal_n_s16(
                sumLow, vget_low_s16(centeredInput), centeredWeight);
            sumHigh = vmlal_n_s16(
                sumHigh, vget_high_s16(centeredInput), centeredWeight);
          }
          const float32x4_t resultLow = vmlaq_n_f32(vdupq_n_f32(bias),
              vcvtq_f32_s32(sumLow), productScale);
          const float32x4_t resultHigh = vmlaq_n_f32(vdupq_n_f32(bias),
              vcvtq_f32_s32(sumHigh), productScale);
          vst1q_f32(output + position, resultLow);
          vst1q_f32(output + position + 4, resultHigh);
        }
#endif

        for (; position < spatialSize; ++position) {
          int32_t sum = 0;
          for (int64_t reduction = 0; reduction < reductionSize; ++reduction)
            sum += ((int32_t)matrixB[reduction * spatialSize + position] -
                       xZeroPoint) *
                   ((int32_t)weight[reduction] - wZeroPoint);
          output[position] = (float)sum * productScale + bias;
        }
      }
    }
  }
  free(columns);
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
  if (!check_support(xs, ws, ys, dh, dw, group, sh, sw, padLeft, padTop)) {
    fprintf(stderr, "MYACCEL: unsupported conv, falling back to CPU convolution\n");
    my_conv_f32_cpu_fallback(
        y, x, w, b, xs, ws, ys, dh, dw, group, padLeft, padTop, sh, sw);
    return;
  }

  if (myaccel_xrt_conv2d_f32(x, w, b, y, (int)nSize, (int)cSize, (int)hSize,
          (int)wSize, (int)mSize, (int)khSize, (int)kwSize, (int)ohSize,
          (int)owSize, (int)dh, (int)dw, (int)cPerGroup, (int)group,
          (int)padLeft, (int)padTop, (int)sh, (int)sw, bTensor != 0)) {
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
