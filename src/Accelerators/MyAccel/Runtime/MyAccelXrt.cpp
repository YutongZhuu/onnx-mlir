#include "MyAccelXrt.h"
#include <cstdlib>
#include <stdio.h>

#ifndef MYACCEL_USE_XRT

extern "C" int myaccel_xrt_conv2d_f32(const float *, const float *,
    const float *, float *, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int) {
  fprintf(stderr, "MYACCEL: XRT support not compiled in\n");
  return 0;
}

#else

#include <chrono>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool envFlagEnabled(const char *name) {
  const char *value = getenv(name);
  return value && value[0] && strcmp(value, "0") != 0;
}

struct ReusableBo {
  std::unique_ptr<xrt::bo> object;
  size_t capacity = 0;
  int groupId = -1;

  bool ensure(xrt::device &device, size_t requiredBytes, int requiredGroupId) {
    if (object && capacity >= requiredBytes && groupId == requiredGroupId)
      return false;

    object =
        std::make_unique<xrt::bo>(device, requiredBytes, requiredGroupId);
    capacity = requiredBytes;
    groupId = requiredGroupId;
    return true;
  }

  xrt::bo &get() { return *object; }
};

struct KernelBufferPool {
  ReusableBo input;
  ReusableBo weight;
  ReusableBo bias;
  ReusableBo output;
};

struct XrtContext {
  xrt::device device;
  xrt::uuid uuid;
  xrt::kernel conv1x1Kernel;
  xrt::kernel conv3x3Kernel;
  KernelBufferPool conv1x1Buffers;
  KernelBufferPool conv3x3Buffers;
  std::mutex executionMutex;

  explicit XrtContext(const char *xclbin)
      : device(0), uuid(device.load_xclbin(xclbin)),
        conv1x1Kernel(device, uuid, "conv1x1_kernel"),
        conv3x3Kernel(device, uuid, "conv3x3_kernel") {}
};

std::unique_ptr<XrtContext> g_ctx;
std::mutex g_contextMutex;

XrtContext *getContext() {
  std::lock_guard<std::mutex> lock(g_contextMutex);
  if (g_ctx)
    return g_ctx.get();

  const char *xclbin = getenv("MYACCEL_XCLBIN");
  if (!xclbin || !xclbin[0]) {
    fprintf(stderr, "MYACCEL: MYACCEL_XCLBIN is not set\n");
    return nullptr;
  }

  fprintf(stderr, "MYACCEL: loading xclbin: %s\n", xclbin);
  const auto start = Clock::now();
  g_ctx = std::make_unique<XrtContext>(xclbin);
  fprintf(stderr,
      "MYACCEL: loaded xclbin and opened both convolution kernels in %.3f ms\n",
      milliseconds(start, Clock::now()));
  return g_ctx.get();
}

} // namespace

extern "C" int myaccel_xrt_conv2d_f32(const float *x, const float *weight,
    const float *bias, float *y, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int kh_size, int kw_size, int oh_size,
    int ow_size, int dilation_h, int dilation_w, int c_per_group, int group,
    int pad_left, int pad_top, int stride_h, int stride_w, int has_bias) {
  try {
    const bool profile = envFlagEnabled("MYACCEL_PROFILE");
    const bool useBufferPool =
        !envFlagEnabled("MYACCEL_DISABLE_BO_POOL");
    const auto totalStart = Clock::now();
    const auto contextStart = Clock::now();
    XrtContext *ctx = getContext();
    const auto contextEnd = Clock::now();
    if (!ctx)
      return 0;
    if (dilation_h != 1 || dilation_w != 1 || group != 1 ||
        c_per_group != c_size)
      return 0;

    xrt::kernel *kernel = nullptr;
    KernelBufferPool *persistentPool = nullptr;
    const char *kernelName = nullptr;
    if (kh_size == 1 && kw_size == 1) {
      kernel = &ctx->conv1x1Kernel;
      persistentPool = &ctx->conv1x1Buffers;
      kernelName = "conv1x1_kernel";
    } else if (kh_size == 3 && kw_size == 3) {
      kernel = &ctx->conv3x3Kernel;
      persistentPool = &ctx->conv3x3Buffers;
      kernelName = "conv3x3_kernel";
    } else {
      return 0;
    }

    size_t xBytes =
        (size_t)n_size * c_size * h_size * input_w_size * sizeof(float);
    size_t wBytes =
        (size_t)m_size * c_per_group * kh_size * kw_size * sizeof(float);
    size_t bBytes = (size_t)(has_bias ? m_size : 1) * sizeof(float);
    size_t yBytes =
        (size_t)n_size * m_size * oh_size * ow_size * sizeof(float);

    fprintf(stderr,
        "MYACCEL: XRT launch preparing bytes={x:%zu,w:%zu,b:%zu,y:%zu}\n",
        xBytes, wBytes, bBytes, yBytes);

    const auto queueStart = Clock::now();
    std::lock_guard<std::mutex> executionLock(ctx->executionMutex);
    const auto queueEnd = Clock::now();

    // When pooling is disabled, these BOs are destroyed at the end of this
    // call. This provides an A/B comparison using the same runtime binary.
    KernelBufferPool transientPool;
    KernelBufferPool &buffers =
        useBufferPool ? *persistentPool : transientPool;

    const auto allocationStart = Clock::now();
    int allocationMask = 0;
    if (buffers.input.ensure(ctx->device, xBytes, kernel->group_id(0)))
      allocationMask |= 1;
    if (buffers.weight.ensure(ctx->device, wBytes, kernel->group_id(1)))
      allocationMask |= 2;
    if (buffers.bias.ensure(ctx->device, bBytes, kernel->group_id(2)))
      allocationMask |= 4;
    if (buffers.output.ensure(ctx->device, yBytes, kernel->group_id(3)))
      allocationMask |= 8;
    const auto allocationEnd = Clock::now();
    fprintf(stderr, "MYACCEL: XRT BO allocation/reuse complete in %.3f ms\n",
        milliseconds(allocationStart, allocationEnd));

    xrt::bo &xBo = buffers.input.get();
    xrt::bo &wBo = buffers.weight.get();
    xrt::bo &bBo = buffers.bias.get();
    xrt::bo &yBo = buffers.output.get();

    const auto xWriteStart = Clock::now();
    xBo.write(x, xBytes, 0);
    const auto xWriteEnd = Clock::now();
    const auto wWriteStart = Clock::now();
    wBo.write(weight, wBytes, 0);
    const auto wWriteEnd = Clock::now();

    float dummyBias = 0.0f;
    const auto bWriteStart = Clock::now();
    bBo.write(has_bias ? bias : &dummyBias, bBytes, 0);
    const auto bWriteEnd = Clock::now();
    fprintf(stderr, "MYACCEL: XRT host writes complete in %.3f ms\n",
        milliseconds(xWriteStart, bWriteEnd));

    const auto xSyncStart = Clock::now();
    xBo.sync(XCL_BO_SYNC_BO_TO_DEVICE, xBytes, 0);
    const auto xSyncEnd = Clock::now();
    const auto wSyncStart = Clock::now();
    wBo.sync(XCL_BO_SYNC_BO_TO_DEVICE, wBytes, 0);
    const auto wSyncEnd = Clock::now();
    const auto bSyncStart = Clock::now();
    bBo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bBytes, 0);
    const auto bSyncEnd = Clock::now();
    fprintf(stderr, "MYACCEL: XRT input sync complete in %.3f ms\n",
        milliseconds(xSyncStart, bSyncEnd));

    Clock::time_point submitEnd;
    Clock::time_point waitStart;
    Clock::time_point waitEnd;
    const auto submitStart = Clock::now();
    fprintf(stderr, "MYACCEL: XRT kernel launch start\n");
    if (kh_size == 1) {
      auto run = (*kernel)(xBo, wBo, bBo, yBo, n_size, c_size, h_size,
          input_w_size, m_size, has_bias);
      submitEnd = Clock::now();
      waitStart = Clock::now();
      run.wait();
      waitEnd = Clock::now();
    } else {
      auto run = (*kernel)(xBo, wBo, bBo, yBo, n_size, c_size, h_size,
          input_w_size, m_size, oh_size, ow_size, pad_left, pad_top,
          stride_h, stride_w, has_bias);
      submitEnd = Clock::now();
      waitStart = Clock::now();
      run.wait();
      waitEnd = Clock::now();
    }
    fprintf(stderr, "MYACCEL: XRT kernel wait complete in %.3f ms\n",
        milliseconds(submitStart, waitEnd));

    const auto ySyncStart = Clock::now();
    yBo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, yBytes, 0);
    const auto ySyncEnd = Clock::now();
    const auto readStart = Clock::now();
    yBo.read(y, yBytes, 0);
    const auto readEnd = Clock::now();
    fprintf(stderr, "MYACCEL: XRT output sync/read complete in %.3f ms\n",
        milliseconds(ySyncStart, readEnd));

    if (profile) {
      fprintf(stderr,
          "MYACCEL_PROFILE kernel=%s shape=%dx%dx%dx%d->%dx%dx%d "
          "bytes=x:%zu,w:%zu,b:%zu,y:%zu pool=%s alloc_mask=0x%x "
          "context=%.3f queue=%.3f alloc=%.3f "
          "x_write=%.3f w_write=%.3f b_write=%.3f "
          "x_sync=%.3f w_sync=%.3f b_sync=%.3f "
          "submit=%.3f wait=%.3f kernel_total=%.3f "
          "y_sync=%.3f read=%.3f total=%.3f ms\n",
          kernelName, n_size, c_size, h_size, input_w_size, m_size, oh_size,
          ow_size, xBytes, wBytes, bBytes, yBytes,
          useBufferPool ? "on" : "off", allocationMask,
          milliseconds(contextStart, contextEnd),
          milliseconds(queueStart, queueEnd),
          milliseconds(allocationStart, allocationEnd),
          milliseconds(xWriteStart, xWriteEnd),
          milliseconds(wWriteStart, wWriteEnd),
          milliseconds(bWriteStart, bWriteEnd),
          milliseconds(xSyncStart, xSyncEnd),
          milliseconds(wSyncStart, wSyncEnd),
          milliseconds(bSyncStart, bSyncEnd),
          milliseconds(submitStart, submitEnd),
          milliseconds(waitStart, waitEnd),
          milliseconds(submitStart, waitEnd),
          milliseconds(ySyncStart, ySyncEnd),
          milliseconds(readStart, readEnd), milliseconds(totalStart, readEnd));
    }

    return 1;
  } catch (const std::exception &e) {
    fprintf(stderr, "MYACCEL: XRT conv failed: %s\n", e.what());
    return 0;
  }
}

#endif
