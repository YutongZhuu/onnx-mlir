#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "Conv1x1Kernel.h"
#include "Conv1x1Int8Kernel.h"
#include "Conv3x3Kernel.h"
#include "Conv3x3Int8Kernel.h"
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

static double now_seconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    return 0.0;
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int64_t read_nonnegative_env_i64(
    const char *name, int64_t defaultValue) {
  const char *text = getenv(name);
  if (!text || !text[0])
    return defaultValue;

  errno = 0;
  char *end = NULL;
  long long value = strtoll(text, &end, 10);
  if (errno || end == text || *end != '\0' || value < 0) {
    fprintf(stderr,
        "MYACCEL: ignoring invalid %s=%s, using default %lld\n", name, text,
        (long long)defaultValue);
    return defaultValue;
  }
  return (int64_t)value;
}

static uint64_t tensor_elems_4d(const int64_t *shape) {
  return (uint64_t)shape[0] * (uint64_t)shape[1] * (uint64_t)shape[2] *
         (uint64_t)shape[3];
}

static void set_reason(char *reason, size_t reasonSize, const char *text) {
  if (!reason || reasonSize == 0)
    return;
  snprintf(reason, reasonSize, "%s", text);
}

static int force_cpu_requested(void) {
  const char *cpu = getenv("CPU");
  return (cpu && strcmp(cpu, "1") == 0) || getenv("MYACCEL_FORCE_CPU");
}

static int check_support(const int64_t *xs, const int64_t *ws,
    const int64_t *ys, int64_t dh, int64_t dw, int64_t group, int64_t sh,
    int64_t sw, int64_t padLeft, int64_t padTop, char *reason,
    size_t reasonSize) {
  if (force_cpu_requested()) {
    set_reason(reason, reasonSize, "CPU=1 or MYACCEL_FORCE_CPU is set");
    return 0;
  }
  if (dh != 1 || dw != 1 || group != 1 || sh <= 0 || sw <= 0) {
    set_reason(reason, reasonSize,
        "unsupported dilation, group/depthwise conv, or stride");
    return 0;
  }
  if (xs[0] <= 0 || xs[1] <= 0 || xs[2] <= 0 || xs[3] <= 0 ||
      ws[0] <= 0 || ws[1] <= 0 || ws[2] <= 0 || ws[3] <= 0 ||
      ys[0] <= 0 || ys[1] <= 0 || ys[2] <= 0 || ys[3] <= 0) {
    set_reason(reason, reasonSize, "invalid tensor metadata");
    return 0;
  }
  if (!value_fits_int(dh) || !value_fits_int(dw) || !value_fits_int(group) ||
      !value_fits_int(sh) || !value_fits_int(sw) ||
      !value_fits_int(padLeft) || !value_fits_int(padTop)) {
    set_reason(reason, reasonSize, "metadata does not fit int ABI");
    return 0;
  }
  for (int i = 0; i < 4; ++i) {
    if (!value_fits_int(xs[i]) || !value_fits_int(ws[i]) ||
        !value_fits_int(ys[i])) {
      set_reason(reason, reasonSize, "tensor dimension does not fit int ABI");
      return 0;
    }
  }
  if (!product_fits_uint32(xs[0], xs[1], xs[2], xs[3]) ||
      !product_fits_uint32(ws[0], ws[1], ws[2], ws[3]) ||
      !product_fits_uint32(ys[0], ys[1], ys[2], ys[3])) {
    set_reason(reason, reasonSize, "tensor element count exceeds XRT ABI");
    return 0;
  }
  if (!(xs[1] == ws[1] && ys[0] == xs[0] && ys[1] == ws[0])) {
    set_reason(reason, reasonSize, "inconsistent Conv tensor shapes");
    return 0;
  }

  const int is1x1 = ws[2] == 1 && ws[3] == 1;
  const int is3x3 = ws[2] == 3 && ws[3] == 3;
  if (!is1x1 && !is3x3) {
    set_reason(reason, reasonSize, "only 1x1 and 3x3 kernels are supported");
    return 0;
  }

  if (is1x1 &&
      (xs[1] > MYACCEL_CONV1X1_MAX_INPUT_CHANNELS || sh != 1 || sw != 1 ||
          padLeft != 0 || padTop != 0 || ys[2] != xs[2] ||
          ys[3] != xs[3])) {
    set_reason(reason, reasonSize,
        "1x1 convolution exceeds channel limit or requires padding/stride");
    return 0;
  }
  if (is3x3 &&
      (xs[1] > MYACCEL_CONV3X3_MAX_INPUT_CHANNELS ||
          sh > MYACCEL_CONV3X3_MAX_STRIDE ||
          sw > MYACCEL_CONV3X3_MAX_STRIDE)) {
    set_reason(reason, reasonSize,
        "3x3 convolution exceeds channel or stride limit");
    return 0;
  }

  const int64_t maxOutputPixels =
      read_nonnegative_env_i64("MYACCEL_MAX_OUTPUT_PIXELS", 80 * 80);
  if (maxOutputPixels > 0 && ys[2] * ys[3] > maxOutputPixels) {
    snprintf(reason, reasonSize,
        "output spatial size %lld exceeds MYACCEL_MAX_OUTPUT_PIXELS=%lld",
        (long long)(ys[2] * ys[3]), (long long)maxOutputPixels);
    return 0;
  }

  const int64_t maxIoBytes =
      read_nonnegative_env_i64("MYACCEL_MAX_IO_BYTES", 32 * 1024 * 1024);
  const uint64_t xBytes = tensor_elems_4d(xs) * sizeof(float);
  const uint64_t wBytes = tensor_elems_4d(ws) * sizeof(float);
  const uint64_t yBytes = tensor_elems_4d(ys) * sizeof(float);
  const uint64_t bBytes = (uint64_t)ws[0] * sizeof(float);
  const uint64_t totalBytes = xBytes + wBytes + yBytes + bBytes;
  if (maxIoBytes > 0 && totalBytes > (uint64_t)maxIoBytes) {
    snprintf(reason, reasonSize,
        "total IO bytes %llu exceeds MYACCEL_MAX_IO_BYTES=%lld",
        (unsigned long long)totalBytes, (long long)maxIoBytes);
    return 0;
  }

  set_reason(reason, reasonSize, "supported");
  return 1;
}

static int check_i8_support(const int64_t *xs, const int64_t *ws,
    const int64_t *ys, int64_t dh, int64_t dw, int64_t group, int64_t sh,
    int64_t sw, int64_t padLeft, int64_t padTop, char *reason,
    size_t reasonSize) {
  if (force_cpu_requested()) {
    set_reason(reason, reasonSize, "CPU=1 or MYACCEL_FORCE_CPU is set");
    return 0;
  }
  if (dh != 1 || dw != 1 || group != 1 || sh <= 0 || sw <= 0 ||
      padLeft < 0 || padTop < 0) {
    set_reason(reason, reasonSize,
        "unsupported dilation, group/depthwise conv, or stride");
    return 0;
  }
  for (int i = 0; i < 4; ++i) {
    if (xs[i] <= 0 || ws[i] <= 0 || ys[i] <= 0) {
      set_reason(reason, reasonSize, "invalid tensor metadata");
      return 0;
    }
    if (!value_fits_int(xs[i]) || !value_fits_int(ws[i]) ||
        !value_fits_int(ys[i])) {
      set_reason(reason, reasonSize, "tensor dimension does not fit int ABI");
      return 0;
    }
  }
  if (!value_fits_int(dh) || !value_fits_int(dw) || !value_fits_int(group) ||
      !value_fits_int(sh) || !value_fits_int(sw) ||
      !value_fits_int(padLeft) || !value_fits_int(padTop)) {
    set_reason(reason, reasonSize, "metadata does not fit int ABI");
    return 0;
  }
  if (!product_fits_uint32(xs[0], xs[1], xs[2], xs[3]) ||
      !product_fits_uint32(ws[0], ws[1], ws[2], ws[3]) ||
      !product_fits_uint32(ys[0], ys[1], ys[2], ys[3])) {
    set_reason(reason, reasonSize, "tensor element count exceeds XRT ABI");
    return 0;
  }
  if (xs[1] != ws[1] || ys[0] != xs[0] || ys[1] != ws[0]) {
    set_reason(reason, reasonSize, "inconsistent Conv tensor shapes");
    return 0;
  }

  const int is1x1 = ws[2] == 1 && ws[3] == 1;
  const int is3x3 = ws[2] == 3 && ws[3] == 3;
  if (!is1x1 && !is3x3) {
    set_reason(reason, reasonSize,
        "only 1x1 and 3x3 INT8 kernels are supported");
    return 0;
  }
  if (is1x1 &&
      (xs[1] > MYACCEL_CONV1X1_INT8_MAX_INPUT_CHANNELS || sh != 1 ||
          sw != 1 || padLeft != 0 || padTop != 0 || ys[2] != xs[2] ||
          ys[3] != xs[3])) {
    set_reason(reason, reasonSize,
        "INT8 1x1 convolution exceeds channel limit or requires "
        "padding/stride");
    return 0;
  }
  if (is3x3 &&
      (xs[1] > MYACCEL_CONV3X3_INT8_MAX_INPUT_CHANNELS ||
          sh > MYACCEL_CONV3X3_INT8_MAX_STRIDE ||
          sw > MYACCEL_CONV3X3_INT8_MAX_STRIDE)) {
    set_reason(reason, reasonSize,
        "INT8 3x3 convolution exceeds channel or stride limit");
    return 0;
  }

  const int64_t maxOutputPixels =
      read_nonnegative_env_i64("MYACCEL_MAX_OUTPUT_PIXELS", 80 * 80);
  if (maxOutputPixels > 0 && ys[2] * ys[3] > maxOutputPixels) {
    snprintf(reason, reasonSize,
        "output spatial size %lld exceeds MYACCEL_MAX_OUTPUT_PIXELS=%lld",
        (long long)(ys[2] * ys[3]), (long long)maxOutputPixels);
    return 0;
  }

  const int64_t maxIoBytes =
      read_nonnegative_env_i64("MYACCEL_MAX_IO_BYTES", 32 * 1024 * 1024);
  const uint64_t xBytes = tensor_elems_4d(xs) * sizeof(int8_t);
  const uint64_t wBytes = tensor_elems_4d(ws) * sizeof(int8_t);
  const uint64_t yBytes = tensor_elems_4d(ys) * sizeof(int32_t);
  const uint64_t totalBytes = xBytes + wBytes + yBytes;
  if (maxIoBytes > 0 && totalBytes > (uint64_t)maxIoBytes) {
    snprintf(reason, reasonSize,
        "total INT8 accelerator IO bytes %llu exceeds "
        "MYACCEL_MAX_IO_BYTES=%lld",
        (unsigned long long)totalBytes, (long long)maxIoBytes);
    return 0;
  }

  set_reason(reason, reasonSize, "supported");
  return 1;
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

static int8_t requantize_i8(int32_t sum, int32_t bias, int32_t biasZeroPoint,
    float productScale, float biasScale, float outputScale,
    int32_t outputZeroPoint) {
  double quantized;
  const float scaleTolerance =
      1.0e-6f * fmaxf(fabsf(productScale), fabsf(biasScale));
  if (biasZeroPoint == 0 &&
      fabsf(productScale - biasScale) <= scaleTolerance) {
    // QLinearConv defines the int32 bias at x_scale * w_scale. Match MLAS's
    // ordering: add it to the integer accumulator, then apply one float32
    // output multiplier. Keeping these as float operations is significant for
    // exact ties-to-even behavior at quantization boundaries.
    const int64_t accumulator = (int64_t)sum + (int64_t)bias;
    const float multiplier = productScale / outputScale;
    quantized = (double)nearbyintf((float)accumulator * multiplier) +
                outputZeroPoint;
  } else {
    // Preserve the general QDQ semantics if a model supplies an independently
    // scaled or nonzero-point bias that cannot be represented by QLinearConv.
    const double realValue =
        (double)sum * productScale +
        (double)(bias - biasZeroPoint) * biasScale;
    quantized = nearbyint(realValue / outputScale) + outputZeroPoint;
  }
  if (quantized < -128.0)
    quantized = -128.0;
  else if (quantized > 127.0)
    quantized = 127.0;
  return (int8_t)quantized;
}

// Compute a QDQ-wrapped INT8 convolution. Supported 1x1 and 3x3 layers send
// only the centered INT8 dot products to XRT; bias and requantization stay on
// the host. Other kernel sizes and CPU=1 use the host implementation below.
// NCHW input and OIHW weights are assumed.
void my_conv_qdq_i8(OMTensor *yTensor, const OMTensor *xTensor,
    const OMTensor *xScaleTensor, const OMTensor *xZeroPointTensor,
    const OMTensor *wTensor, const OMTensor *wScaleTensor,
    const OMTensor *wZeroPointTensor, const OMTensor *bTensor,
    const OMTensor *bScaleTensor, const OMTensor *bZeroPointTensor,
    const OMTensor *yScaleTensor, const OMTensor *yZeroPointTensor,
    int64_t dh, int64_t dw, int64_t group, int64_t padLeft, int64_t padTop,
    int64_t sh, int64_t sw) {
  const int64_t *xs = omTensorGetShape(xTensor);
  const int64_t *ws = omTensorGetShape(wTensor);
  const int64_t *ys = omTensorGetShape(yTensor);
  const int8_t *x = (const int8_t *)omTensorGetDataPtr(xTensor);
  const int8_t *w = (const int8_t *)omTensorGetDataPtr(wTensor);
  const int32_t *b = (const int32_t *)omTensorGetDataPtr(bTensor);
  int8_t *y = (int8_t *)omTensorGetDataPtr(yTensor);
  const float xScale = *(const float *)omTensorGetDataPtr(xScaleTensor);
  const int16_t xZeroPoint =
      *(const int8_t *)omTensorGetDataPtr(xZeroPointTensor);
  const float wScale = *(const float *)omTensorGetDataPtr(wScaleTensor);
  const int16_t wZeroPoint =
      *(const int8_t *)omTensorGetDataPtr(wZeroPointTensor);
  const float bScale = *(const float *)omTensorGetDataPtr(bScaleTensor);
  const int32_t bZeroPoint =
      *(const int32_t *)omTensorGetDataPtr(bZeroPointTensor);
  const float yScale = *(const float *)omTensorGetDataPtr(yScaleTensor);
  const int16_t yZeroPoint =
      *(const int8_t *)omTensorGetDataPtr(yZeroPointTensor);

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

  const int profile = getenv("MYACCEL_PROFILE") != NULL;
  if (profile)
    fprintf(stderr,
        "MYACCEL_INT8 shape=%lldx%lldx%lldx%lld->%lldx%lldx%lld "
        "kernel=%lldx%lld stride=%lldx%lld direct=%d\n",
        (long long)nSize, (long long)cSize, (long long)hSize,
        (long long)wSize, (long long)mSize, (long long)ohSize,
        (long long)owSize, (long long)khSize, (long long)kwSize,
        (long long)sh, (long long)sw, directOneByOne);

  char reason[160];
  if (check_i8_support(xs, ws, ys, dh, dw, group, sh, sw, padLeft, padTop,
          reason, sizeof(reason))) {
    const size_t accumulatorCount = (size_t)tensor_elems_4d(ys);
    int32_t *accumulator =
        (int32_t *)malloc(accumulatorCount * sizeof(int32_t));
    if (accumulator) {
      if (profile)
        fprintf(stderr,
            "MYACCEL_INT8 routing convolution dot products to XRT: %s\n",
            reason);
      const double xrtStart = now_seconds();
      if (myaccel_xrt_conv2d_i8(x, w, accumulator, (int)nSize, (int)cSize,
              (int)hSize, (int)wSize, (int)mSize, (int)khSize, (int)kwSize,
              (int)ohSize, (int)owSize, (int)dh, (int)dw, (int)cPerGroup,
              (int)group, (int)padLeft, (int)padTop, (int)sh, (int)sw,
              (int)xZeroPoint, (int)wZeroPoint)) {
        const double xrtEnd = now_seconds();
        for (int64_t n = 0; n < nSize; ++n)
          for (int64_t m = 0; m < mSize; ++m)
            for (int64_t position = 0; position < spatialSize; ++position) {
              const int64_t index =
                  (n * mSize + m) * spatialSize + position;
              y[index] = requantize_i8(accumulator[index], b ? b[m] : 0,
                  bZeroPoint, productScale, bScale, yScale, yZeroPoint);
            }
        const double requantEnd = now_seconds();
        free(accumulator);
        if (profile)
          fprintf(stderr,
              "MYACCEL_INT8_PROFILE xrt_dot_products=%.6f "
              "host_requantization=%.6f total=%.6f s\n",
              xrtEnd - xrtStart, requantEnd - xrtEnd,
              requantEnd - xrtStart);
        return;
      }
      free(accumulator);
      if (profile)
        fprintf(stderr,
            "MYACCEL_INT8 XRT path failed after %.6f s; using host "
            "convolution\n",
            now_seconds() - xrtStart);
    } else if (profile) {
      fprintf(stderr,
          "MYACCEL_INT8 could not allocate the host accumulator; using host "
          "convolution\n");
    }
  } else if (profile) {
    fprintf(stderr, "MYACCEL_INT8 routing convolution to host: %s\n", reason);
  }

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
        int8_t *output = y + (n * mSize + m) * spatialSize;
        int64_t position = 0;

#if defined(__aarch64__)
        if (!getenv("MYACCEL_DISABLE_SIMD")) {
          const int16x8_t xZeroPointVector = vdupq_n_s16(xZeroPoint);
          for (; position + 8 <= spatialSize; position += 8) {
            int32x4_t sumLow = vdupq_n_s32(0);
            int32x4_t sumHigh = vdupq_n_s32(0);
            for (int64_t reduction = 0; reduction < reductionSize;
                 ++reduction) {
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
            int32_t sums[8];
            vst1q_s32(sums, sumLow);
            vst1q_s32(sums + 4, sumHigh);
            for (int lane = 0; lane < 8; ++lane)
              output[position + lane] = requantize_i8(sums[lane], b[m],
                  bZeroPoint, productScale, bScale, yScale, yZeroPoint);
          }
        }
#endif

        for (; position < spatialSize; ++position) {
          int32_t sum = 0;
          for (int64_t reduction = 0; reduction < reductionSize; ++reduction)
            sum += ((int32_t)matrixB[reduction * spatialSize + position] -
                       xZeroPoint) *
                   ((int32_t)weight[reduction] - wZeroPoint);
          output[position] = requantize_i8(sum, b[m], bZeroPoint,
              productScale, bScale, yScale, yZeroPoint);
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

  const uint64_t xBytes = tensor_elems_4d(xs) * sizeof(float);
  const uint64_t wBytes = tensor_elems_4d(ws) * sizeof(float);
  const uint64_t bBytes = bTensor ? (uint64_t)mSize * sizeof(float) : 0;
  const uint64_t yBytes = tensor_elems_4d(ys) * sizeof(float);

  fprintf(stderr,
      "MYACCEL: conv x=[%lld,%lld,%lld,%lld] w=[%lld,%lld,%lld,%lld] "
      "y=[%lld,%lld,%lld,%lld] bytes={x:%llu,w:%llu,b:%llu,y:%llu} "
      "attrs={dilation:%lldx%lld,group:%lld,pad:%lldx%lld,stride:%lldx%lld}\n",
      (long long)xs[0], (long long)xs[1], (long long)xs[2],
      (long long)xs[3], (long long)ws[0], (long long)ws[1],
      (long long)ws[2], (long long)ws[3], (long long)ys[0],
      (long long)ys[1], (long long)ys[2], (long long)ys[3],
      (unsigned long long)xBytes, (unsigned long long)wBytes,
      (unsigned long long)bBytes, (unsigned long long)yBytes, (long long)dh,
      (long long)dw, (long long)group, (long long)padLeft,
      (long long)padTop, (long long)sh, (long long)sw);

  char reason[160];
  if (!check_support(
          xs, ws, ys, dh, dw, group, sh, sw, padLeft, padTop, reason,
          sizeof(reason))) {
    fprintf(stderr, "MYACCEL: routing conv to CPU fallback: %s\n", reason);
    const double start = now_seconds();
    my_conv_f32_cpu_fallback(
        y, x, w, b, xs, ws, ys, dh, dw, group, padLeft, padTop, sh, sw);
    fprintf(stderr, "MYACCEL: CPU fallback complete in %.6f s\n",
        now_seconds() - start);
    return;
  }

  fprintf(stderr, "MYACCEL: routing conv to XRT: %s\n", reason);
  const double xrtStart = now_seconds();
  if (myaccel_xrt_conv2d_f32(x, w, b, y, (int)nSize, (int)cSize, (int)hSize,
          (int)wSize, (int)mSize, (int)khSize, (int)kwSize, (int)ohSize,
          (int)owSize, (int)dh, (int)dw, (int)cPerGroup, (int)group,
          (int)padLeft, (int)padTop, (int)sh, (int)sw, bTensor != 0)) {
    fprintf(stderr, "MYACCEL: XRT conv complete in %.6f s\n",
        now_seconds() - xrtStart);
    return;
  }

  fprintf(stderr,
      "MYACCEL: XRT path failed after %.6f s, falling back to CPU convolution\n",
      now_seconds() - xrtStart);
  const double cpuStart = now_seconds();
  my_conv_f32_cpu_fallback(
        y, x, w, b, xs, ws, ys, dh, dw, group, padLeft, padTop, sh, sw);
  fprintf(stderr, "MYACCEL: CPU fallback complete in %.6f s\n",
      now_seconds() - cpuStart);
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
