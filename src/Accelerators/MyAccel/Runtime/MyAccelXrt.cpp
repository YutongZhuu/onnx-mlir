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

#include <exception>
#include <memory>
#include <string>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// real XRT implementation goes here

namespace {

struct XrtContext {
  xrt::device device;
  xrt::uuid uuid;
  xrt::kernel kernel;

  explicit XrtContext(const char *xclbin)
      : device(0), uuid(device.load_xclbin(xclbin)),
        kernel(device, uuid, "conv2d_kernel") {}
};

std::unique_ptr<XrtContext> g_ctx;

XrtContext *getContext() {
  if (g_ctx)
    return g_ctx.get();

  const char *xclbin = getenv("MYACCEL_XCLBIN");
  if (!xclbin || !xclbin[0]) {
    fprintf(stderr, "MYACCEL: MYACCEL_XCLBIN is not set\n");
    return nullptr;
  }

  g_ctx = std::make_unique<XrtContext>(xclbin);
  return g_ctx.get();
}

} // namespace

extern "C" int myaccel_xrt_conv2d_f32(const float *x, const float *weight,
    const float *bias, float *y, int n_size, int c_size, int h_size,
    int input_w_size, int m_size, int kh_size, int kw_size, int oh_size,
    int ow_size, int dilation_h, int dilation_w, int c_per_group, int group,
    int pad_left, int pad_top, int stride_h, int stride_w, int has_bias) {
  try {
      
    XrtContext *ctx = getContext();
    if (!ctx)
      return 0;
      
    size_t xBytes = (size_t)n_size * c_size * h_size * input_w_size * sizeof(float);
    size_t wBytes = (size_t)m_size * c_per_group * kh_size * kw_size * sizeof(float);
    size_t bBytes = (size_t)(has_bias ? m_size : 1) * sizeof(float);
    size_t yBytes = (size_t)n_size * m_size * oh_size * ow_size * sizeof(float);
    
    xrt::bo xBo(ctx->device, xBytes, ctx->kernel.group_id(0));
    xrt::bo wBo(ctx->device, wBytes, ctx->kernel.group_id(1));
    xrt::bo bBo(ctx->device, bBytes, ctx->kernel.group_id(2));
    xrt::bo yBo(ctx->device, yBytes, ctx->kernel.group_id(3));
    
    xBo.write(x, xBytes, 0);
    wBo.write(weight, wBytes, 0);
    
    float dummyBias = 0.0f;
    bBo.write(has_bias ? bias : &dummyBias, bBytes, 0);
    
    xBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    wBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    auto run = ctx->kernel(xBo, wBo, bBo, yBo,
      n_size, c_size, h_size, input_w_size,
      m_size, kh_size, kw_size, oh_size, ow_size,
      dilation_h, dilation_w, c_per_group, group,
      pad_left, pad_top, stride_h, stride_w, has_bias);
    run.wait();
    
    yBo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    yBo.read(y, yBytes, 0);

    return 1;
  } catch (const std::exception &e) {
    fprintf(stderr, "MYACCEL: XRT conv failed: %s\n", e.what());
    return 0;
  }
}



#endif
