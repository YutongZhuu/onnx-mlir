#include "MyAccelXrt.h"
#include "Conv1x1Int8Kernel.h"
#include "Conv3x3Int8Kernel.h"
#include "Conv6x6StemInt8Kernel.h"
#include <cstdlib>
#include <stdio.h>

#ifndef MYACCEL_USE_XRT

extern "C" int myaccel_xrt_conv2d_f32(const float *, const float *,
    const float *, float *, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int) {
  fprintf(stderr, "MYACCEL: XRT support not compiled in\n");
  return 0;
}

extern "C" int myaccel_xrt_conv2d_i8(const int8_t *, const int8_t *,
    const int32_t *, int8_t *, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, uint32_t, int) {
  fprintf(stderr, "MYACCEL: XRT support not compiled in\n");
  return 0;
}

#else

#include <chrono>
#include <climits>
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

unsigned int envPositiveUInt(
    const char *name, unsigned int fallback) {
  const char *value = getenv(name);
  if (!value || !value[0])
    return fallback;

  char *end = nullptr;
  const unsigned long parsed = strtoul(value, &end, 10);
  if (!end || *end != '\0' || parsed == 0 || parsed > UINT_MAX) {
    fprintf(stderr, "MYACCEL: ignoring invalid %s=%s\n", name, value);
    return fallback;
  }
  return (unsigned int)parsed;
}

const char *commandStateName(ert_cmd_state state) {
  switch (state) {
  case ERT_CMD_STATE_NEW:
    return "NEW";
  case ERT_CMD_STATE_QUEUED:
    return "QUEUED";
  case ERT_CMD_STATE_RUNNING:
    return "RUNNING";
  case ERT_CMD_STATE_COMPLETED:
    return "COMPLETED";
  case ERT_CMD_STATE_ERROR:
    return "ERROR";
  case ERT_CMD_STATE_ABORT:
    return "ABORT";
  case ERT_CMD_STATE_SUBMITTED:
    return "SUBMITTED";
  case ERT_CMD_STATE_TIMEOUT:
    return "TIMEOUT";
  case ERT_CMD_STATE_NORESPONSE:
    return "NORESPONSE";
  case ERT_CMD_STATE_SKERROR:
    return "SKERROR";
  case ERT_CMD_STATE_SKCRASHED:
    return "SKCRASHED";
  default:
    return "UNKNOWN";
  }
}

ert_cmd_state waitForRun(xrt::run &run, unsigned int timeoutMs,
    const char *kernelName) {
  const ert_cmd_state state = run.wait(timeoutMs);
  if (state != ERT_CMD_STATE_TIMEOUT)
    return state;

  try {
    const ert_cmd_state abortState = run.abort();
    fprintf(stderr,
        "MYACCEL: %s timed out after %u ms; abort returned %s(%d)\n",
        kernelName, timeoutMs, commandStateName(abortState),
        (int)abortState);
  } catch (const std::exception &error) {
    fprintf(stderr,
        "MYACCEL: %s timed out after %u ms; abort failed: %s\n",
        kernelName, timeoutMs, error.what());
  }
  return state;
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
  std::unique_ptr<xrt::kernel> conv1x1Kernel;
  std::unique_ptr<xrt::kernel> conv3x3Kernel;
  std::unique_ptr<xrt::kernel> conv1x1I8Kernel;
  std::unique_ptr<xrt::kernel> conv3x3I8Kernel;
  std::unique_ptr<xrt::kernel> conv6x6StemI8Kernel;
  KernelBufferPool conv1x1Buffers;
  KernelBufferPool conv3x3Buffers;
  KernelBufferPool conv1x1I8Buffers;
  KernelBufferPool conv3x3I8Buffers;
  KernelBufferPool conv6x6StemI8Buffers;
  std::mutex executionMutex;

  explicit XrtContext(const char *xclbin)
      : device(0), uuid(device.load_xclbin(xclbin)) {}
};

std::unique_ptr<XrtContext> g_ctx;
std::mutex g_contextMutex;

void releaseContextAtExit() {
  std::lock_guard<std::mutex> lock(g_contextMutex);
  g_ctx.reset();
}

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
  static const bool cleanupRegistered = []() {
    if (std::atexit(releaseContextAtExit) != 0) {
      fprintf(stderr,
          "MYACCEL: warning: could not register early XRT cleanup\n");
      return false;
    }
    return true;
  }();
  (void)cleanupRegistered;
  fprintf(stderr, "MYACCEL: loaded xclbin in %.3f ms\n",
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

    std::unique_ptr<xrt::kernel> *kernelSlot = nullptr;
    KernelBufferPool *persistentPool = nullptr;
    const char *kernelName = nullptr;
    if (kh_size == 1 && kw_size == 1) {
      kernelSlot = &ctx->conv1x1Kernel;
      persistentPool = &ctx->conv1x1Buffers;
      kernelName = "conv1x1_kernel";
    } else if (kh_size == 3 && kw_size == 3) {
      kernelSlot = &ctx->conv3x3Kernel;
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
    const auto kernelOpenStart = Clock::now();
    if (!*kernelSlot)
      *kernelSlot =
          std::make_unique<xrt::kernel>(ctx->device, ctx->uuid, kernelName);
    xrt::kernel *kernel = kernelSlot->get();
    const auto kernelOpenEnd = Clock::now();

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
          "context=%.3f queue=%.3f kernel_open=%.3f alloc=%.3f "
          "x_write=%.3f w_write=%.3f b_write=%.3f "
          "x_sync=%.3f w_sync=%.3f b_sync=%.3f "
          "submit=%.3f wait=%.3f kernel_total=%.3f "
          "y_sync=%.3f read=%.3f total=%.3f ms\n",
          kernelName, n_size, c_size, h_size, input_w_size, m_size, oh_size,
          ow_size, xBytes, wBytes, bBytes, yBytes,
          useBufferPool ? "on" : "off", allocationMask,
          milliseconds(contextStart, contextEnd),
          milliseconds(queueStart, queueEnd),
          milliseconds(kernelOpenStart, kernelOpenEnd),
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

extern "C" int myaccel_xrt_conv2d_i8(const int8_t *x, const int8_t *weight,
    const int32_t *bias, int8_t *y, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int kh_size, int kw_size, int oh_size,
    int ow_size, int dilation_h, int dilation_w, int c_per_group, int group,
    int pad_left, int pad_top, int stride_h, int stride_w, int x_zero_point,
    int w_zero_point, uint32_t requant_multiplier_bits,
    int output_zero_point) {
  try {
    const bool profile = envFlagEnabled("MYACCEL_PROFILE");
    const bool useBufferPool =
        !envFlagEnabled("MYACCEL_DISABLE_BO_POOL");
    const unsigned int runTimeoutMs =
        envPositiveUInt("MYACCEL_XRT_RUN_TIMEOUT_MS", 30000);
    const auto totalStart = Clock::now();
    if (!x || !weight || !bias || !y || n_size <= 0 || c_size <= 0 ||
        h_size <= 0 || input_w_size <= 0 || m_size <= 0 || oh_size <= 0 ||
        ow_size <= 0)
      return 0;
    if (dilation_h != 1 || dilation_w != 1 || group != 1 ||
        c_per_group != c_size)
      return 0;

    const bool is1x1 = kh_size == 1 && kw_size == 1 &&
        c_size <= MYACCEL_CONV1X1_INT8_MAX_INPUT_CHANNELS &&
        pad_left == 0 && pad_top == 0 && stride_h == 1 && stride_w == 1 &&
        oh_size == h_size && ow_size == input_w_size && c_size % 4 == 0 &&
        ((uint64_t)h_size * input_w_size) % 4 == 0;
    const bool is3x3 = kh_size == 3 && kw_size == 3 &&
        c_size <= MYACCEL_CONV3X3_INT8_MAX_INPUT_CHANNELS &&
        pad_left >= 0 && pad_top >= 0 &&
        stride_h > 0 && stride_h <= MYACCEL_CONV3X3_INT8_MAX_STRIDE &&
        stride_w > 0 && stride_w <= MYACCEL_CONV3X3_INT8_MAX_STRIDE &&
        c_size % 4 == 0 && input_w_size % 4 == 0 && ow_size % 4 == 0;
    const bool is6x6 = envFlagEnabled("MYACCEL_ENABLE_6X6_STEM") &&
        kh_size == 6 && kw_size == 6 &&
        c_size <= MYACCEL_CONV6X6_STEM_INT8_MAX_INPUT_CHANNELS &&
        pad_left >= 0 && pad_top >= 0 && stride_h > 0 &&
        stride_h <= MYACCEL_CONV6X6_STEM_INT8_MAX_STRIDE && stride_w > 0 &&
        stride_w <= MYACCEL_CONV6X6_STEM_INT8_MAX_STRIDE &&
        input_w_size % 4 == 0 && ow_size % 4 == 0;
    if (!is1x1 && !is3x3 && !is6x6)
      return 0;

    const auto contextStart = Clock::now();
    XrtContext *ctx = getContext();
    const auto contextEnd = Clock::now();
    if (!ctx)
      return 0;

    std::unique_ptr<xrt::kernel> *kernelSlot = nullptr;
    KernelBufferPool *persistentPool = nullptr;
    const char *kernelName = nullptr;
    if (is1x1) {
      kernelSlot = &ctx->conv1x1I8Kernel;
      persistentPool = &ctx->conv1x1I8Buffers;
      kernelName = "conv1x1_i8_kernel";
    } else if (is3x3) {
      kernelSlot = &ctx->conv3x3I8Kernel;
      persistentPool = &ctx->conv3x3I8Buffers;
      kernelName = "conv3x3_i8_kernel";
    } else {
      kernelSlot = &ctx->conv6x6StemI8Kernel;
      persistentPool = &ctx->conv6x6StemI8Buffers;
      kernelName = "conv6x6_stem_i8_kernel";
    }

    size_t xBytes =
        (size_t)n_size * c_size * h_size * input_w_size * sizeof(int8_t);
    size_t wBytes = (size_t)m_size * c_per_group * kh_size * kw_size *
                    sizeof(int8_t);
    size_t bBytes = (size_t)m_size * sizeof(int32_t);
    size_t yBytes = (size_t)n_size * m_size * oh_size * ow_size * sizeof(int8_t);

    const auto queueStart = Clock::now();
    std::lock_guard<std::mutex> executionLock(ctx->executionMutex);
    const auto queueEnd = Clock::now();
    const auto kernelOpenStart = Clock::now();
    if (!*kernelSlot)
      *kernelSlot =
          std::make_unique<xrt::kernel>(ctx->device, ctx->uuid, kernelName);
    xrt::kernel *kernel = kernelSlot->get();
    const auto kernelOpenEnd = Clock::now();

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
    const auto bWriteStart = Clock::now();
    bBo.write(bias, bBytes, 0);
    const auto bWriteEnd = Clock::now();

    const auto xSyncStart = Clock::now();
    xBo.sync(XCL_BO_SYNC_BO_TO_DEVICE, xBytes, 0);
    const auto xSyncEnd = Clock::now();
    const auto wSyncStart = Clock::now();
    wBo.sync(XCL_BO_SYNC_BO_TO_DEVICE, wBytes, 0);
    const auto wSyncEnd = Clock::now();
    const auto bSyncStart = Clock::now();
    bBo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bBytes, 0);
    const auto bSyncEnd = Clock::now();

    const auto submitStart = Clock::now();
    Clock::time_point submitEnd;
    Clock::time_point waitStart;
    Clock::time_point waitEnd;
    ert_cmd_state runState = ERT_CMD_STATE_NEW;
    if (is1x1) {
      auto run = (*kernel)(xBo, wBo, bBo, yBo, n_size, c_size, h_size,
          input_w_size, m_size, x_zero_point, w_zero_point,
          requant_multiplier_bits, output_zero_point);
      submitEnd = Clock::now();
      waitStart = Clock::now();
      runState = waitForRun(run, runTimeoutMs, kernelName);
      waitEnd = Clock::now();
    } else {
      auto run = (*kernel)(xBo, wBo, bBo, yBo, n_size, c_size, h_size,
          input_w_size, m_size, oh_size, ow_size, pad_left, pad_top, stride_h,
          stride_w, x_zero_point, w_zero_point, requant_multiplier_bits,
          output_zero_point);
      submitEnd = Clock::now();
      waitStart = Clock::now();
      runState = waitForRun(run, runTimeoutMs, kernelName);
      waitEnd = Clock::now();
    }

    if (runState != ERT_CMD_STATE_COMPLETED) {
      fprintf(stderr, "MYACCEL: %s returned state=%s(%d); using host "
                      "fallback\n",
          kernelName, commandStateName(runState), (int)runState);
      return 0;
    }

    const auto ySyncStart = Clock::now();
    yBo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, yBytes, 0);
    const auto ySyncEnd = Clock::now();
    const auto readStart = Clock::now();
    yBo.read(y, yBytes, 0);
    const auto readEnd = Clock::now();

    if (profile) {
      fprintf(stderr,
          "MYACCEL_PROFILE kernel=%s "
          "shape=%dx%dx%dx%d->%dx%dx%d kernel_shape=%dx%d "
          "bytes=x:%zu,w:%zu,b:%zu,y:%zu pool=%s alloc_mask=0x%x "
          "context=%.3f queue=%.3f kernel_open=%.3f alloc=%.3f "
          "x_write=%.3f w_write=%.3f b_write=%.3f "
          "x_sync=%.3f w_sync=%.3f b_sync=%.3f "
          "submit=%.3f wait=%.3f kernel_total=%.3f "
          "y_sync=%.3f read=%.3f total=%.3f ms\n",
          kernelName, n_size, c_size, h_size, input_w_size, m_size, oh_size,
          ow_size, kh_size, kw_size, xBytes, wBytes, bBytes, yBytes,
          useBufferPool ? "on" : "off", allocationMask,
          milliseconds(contextStart, contextEnd),
          milliseconds(queueStart, queueEnd),
          milliseconds(kernelOpenStart, kernelOpenEnd),
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
    fprintf(stderr, "MYACCEL: XRT INT8 conv failed: %s\n", e.what());
    return 0;
  }
}

#endif
