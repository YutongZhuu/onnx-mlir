#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

struct ConvCase {
  const char *name;
  int n;
  int c;
  int h;
  int w;
  int m;
  int kh;
  int kw;
  int padTop;
  int padLeft;
  int strideH;
  int strideW;
};

void timeoutHandler(int) {
  const char msg[] =
      "FAIL timeout: host_xrt_conv_test exceeded HOST_XRT_TIMEOUT_SEC\n";
  (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
  _exit(124);
}

unsigned int envUInt(const char *name, unsigned int fallback) {
  const char *value = std::getenv(name);
  if (!value || !*value)
    return fallback;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (!end || *end != '\0' || parsed > 0xffffffffUL) {
    std::fprintf(stderr, "warning: ignoring invalid %s=%s\n", name, value);
    return fallback;
  }
  return (unsigned int)parsed;
}

const char *stateName(ert_cmd_state state) {
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

void step(const char *message) {
  std::printf("[host-xrt-test] %s\n", message);
  std::fflush(stdout);
}

bool fileExists(const char *path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

bool deviceNodeExists() {
  return access("/dev/dri/renderD128", R_OK | W_OK) == 0 ||
         access("/dev/zocl", R_OK | W_OK) == 0 ||
         access("/dev/xocl", R_OK | W_OK) == 0;
}

std::vector<float> makeSequence(size_t elems, float scale, float offset) {
  std::vector<float> values(elems);
  for (size_t i = 0; i < elems; ++i)
    values[i] = (float)((int)(i % 17) - 8) * scale + offset;
  return values;
}

void referenceConv(const ConvCase &tc, const std::vector<float> &x,
    const std::vector<float> &weight, const std::vector<float> &bias,
    std::vector<float> &y) {
  const int oh = (tc.h + 2 * tc.padTop - tc.kh) / tc.strideH + 1;
  const int ow = (tc.w + 2 * tc.padLeft - tc.kw) / tc.strideW + 1;

  for (int n = 0; n < tc.n; ++n)
    for (int m = 0; m < tc.m; ++m)
      for (int outH = 0; outH < oh; ++outH)
        for (int outW = 0; outW < ow; ++outW) {
          float sum = bias[m];
          for (int c = 0; c < tc.c; ++c)
            for (int kh = 0; kh < tc.kh; ++kh)
              for (int kw = 0; kw < tc.kw; ++kw) {
                const int inH = outH * tc.strideH + kh - tc.padTop;
                const int inW = outW * tc.strideW + kw - tc.padLeft;
                if (inH < 0 || inH >= tc.h || inW < 0 || inW >= tc.w)
                  continue;

                const size_t xIndex =
                    ((size_t)(n * tc.c + c) * tc.h + inH) * tc.w + inW;
                const size_t wIndex =
                    ((size_t)(m * tc.c + c) * tc.kh + kh) * tc.kw + kw;
                sum += x[xIndex] * weight[wIndex];
              }

          const size_t yIndex =
              ((size_t)(n * tc.m + m) * oh + outH) * ow + outW;
          y[yIndex] = sum;
        }
}

bool runCase(xrt::device &device, xrt::kernel &kernel, const ConvCase &tc,
    unsigned int runTimeoutMs) {
  const int oh = (tc.h + 2 * tc.padTop - tc.kh) / tc.strideH + 1;
  const int ow = (tc.w + 2 * tc.padLeft - tc.kw) / tc.strideW + 1;
  const size_t xElems = (size_t)tc.n * tc.c * tc.h * tc.w;
  const size_t wElems = (size_t)tc.m * tc.c * tc.kh * tc.kw;
  const size_t bElems = (size_t)tc.m;
  const size_t yElems = (size_t)tc.n * tc.m * oh * ow;
  const size_t xBytes = xElems * sizeof(float);
  const size_t wBytes = wElems * sizeof(float);
  const size_t bBytes = bElems * sizeof(float);
  const size_t yBytes = yElems * sizeof(float);

  std::vector<float> x = makeSequence(xElems, 0.125f, 0.25f);
  std::vector<float> weight = makeSequence(wElems, 0.0625f, -0.125f);
  std::vector<float> bias = makeSequence(bElems, 0.25f, 0.5f);
  std::vector<float> expected(yElems, 0.0f);
  std::vector<float> actual(yElems, 0.0f);
  referenceConv(tc, x, weight, bias, expected);

  if (std::string(tc.name) == "tiny") {
    std::fill(x.begin(), x.end(), 0.0f);
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = (float)(i + 1);
    std::fill(weight.begin(), weight.end(), 1.0f);
    std::fill(bias.begin(), bias.end(), 0.5f);
    std::fill(expected.begin(), expected.end(), 0.0f);
    referenceConv(tc, x, weight, bias, expected);
  }

  std::printf("[host-xrt-test] case=%s shape x=[%d,%d,%d,%d] "
              "w=[%d,%d,%d,%d] "
              "y=[%d,%d,%d,%d] bytes={x:%zu,w:%zu,b:%zu,y:%zu}\n",
      tc.name, tc.n, tc.c, tc.h, tc.w, tc.m, tc.c, tc.kh, tc.kw, tc.n, tc.m,
      oh, ow, xBytes, wBytes, bBytes, yBytes);
  std::fflush(stdout);

  step("allocating XRT buffer objects");
  xrt::bo xBo(device, xBytes, kernel.group_id(0));
  xrt::bo wBo(device, wBytes, kernel.group_id(1));
  xrt::bo bBo(device, bBytes, kernel.group_id(2));
  xrt::bo yBo(device, yBytes, kernel.group_id(3));

  step("writing and syncing inputs to device");
  xBo.write(x.data(), xBytes, 0);
  wBo.write(weight.data(), wBytes, 0);
  bBo.write(bias.data(), bBytes, 0);
  xBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  wBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  step("launching kernel");
  auto run = kernel(xBo, wBo, bBo, yBo, tc.n, tc.c, tc.h, tc.w, tc.m, tc.kh,
      tc.kw, oh, ow, 1, 1, tc.c, 1, tc.padLeft, tc.padTop, tc.strideH,
      tc.strideW, 1);
  std::printf("[host-xrt-test] waiting for kernel, timeout_ms=%u\n",
      runTimeoutMs);
  std::fflush(stdout);
  ert_cmd_state state = run.wait(runTimeoutMs);
  std::printf("[host-xrt-test] kernel wait returned state=%s(%d)\n",
      stateName(state), (int)state);
  std::fflush(stdout);
  if (state == ERT_CMD_STATE_TIMEOUT) {
    std::fprintf(stderr, "FAIL case=%s timed out; aborting run\n", tc.name);
    std::fflush(stderr);
    ert_cmd_state abortState = run.abort();
    std::fprintf(stderr, "FAIL case=%s abort returned state=%s(%d)\n", tc.name,
        stateName(abortState), (int)abortState);
    return false;
  }
  if (state != ERT_CMD_STATE_COMPLETED) {
    std::fprintf(stderr, "FAIL case=%s kernel returned state=%s(%d)\n", tc.name,
        stateName(state), (int)state);
    return false;
  }

  step("syncing output from device");
  yBo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  yBo.read(actual.data(), yBytes, 0);

  if (std::string(tc.name) == "tiny") {
    for (size_t i = 0; i < yElems; ++i)
      std::printf("  y[%zu] = %.1f (expected %.1f)\n", i, actual[i],
          expected[i]);
  }

  float maxAbsDiff = 0.0f;
  size_t maxIndex = 0;
  for (size_t i = 0; i < yElems; ++i) {
    const float diff = std::fabs(actual[i] - expected[i]);
    if (diff > maxAbsDiff) {
      maxAbsDiff = diff;
      maxIndex = i;
    }
  }

  const float tolerance = 1e-4f;
  if (maxAbsDiff > tolerance) {
    std::printf("FAIL case=%s max_abs_diff=%g index=%zu actual=%g expected=%g\n",
        tc.name, maxAbsDiff, maxIndex, actual[maxIndex], expected[maxIndex]);
    return false;
  }

  std::printf("PASS case=%s max_abs_diff=%g\n", tc.name, maxAbsDiff);
  return true;
}

void usage(const char *argv0) {
  std::fprintf(stderr,
      "usage: %s <conv2d_kernel.xclbin> [--medium|--all|--probe-only]\n"
      "env: HOST_XRT_TIMEOUT_SEC=30 HOST_XRT_RUN_TIMEOUT_MS=5000\n",
      argv0);
}

} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);

  if (argc < 2 || argc > 3) {
    usage(argv[0]);
    return 2;
  }

  const std::string mode = argc == 3 ? argv[2] : "";
  if (!mode.empty() && mode != "--medium" && mode != "--all" &&
      mode != "--probe-only") {
    usage(argv[0]);
    return 2;
  }

  const unsigned int timeoutSec = envUInt("HOST_XRT_TIMEOUT_SEC", 30);
  const unsigned int runTimeoutMs = envUInt("HOST_XRT_RUN_TIMEOUT_MS", 5000);
  std::signal(SIGALRM, timeoutHandler);
  if (timeoutSec > 0)
    alarm(timeoutSec);

  if (!fileExists(argv[1])) {
    std::fprintf(stderr, "FAIL preflight: xclbin does not exist: %s\n",
        argv[1]);
    return 2;
  }
  if (!deviceNodeExists()) {
    std::fprintf(stderr,
        "FAIL preflight: no usable XRT device node found "
        "(/dev/dri/renderD128, /dev/zocl, or /dev/xocl)\n");
    return 2;
  }

  try {
    step("opening XRT device 0");
    xrt::device device(0);
    step("loading xclbin");
    xrt::uuid uuid = device.load_xclbin(argv[1]);
    step("opening kernel conv2d_kernel");
    xrt::kernel kernel(device, uuid, "conv2d_kernel");
    step("XRT setup completed");
    if (mode == "--probe-only") {
      step("probe-only requested; exiting before kernel launch");
      alarm(0);
      return 0;
    }

    const ConvCase tiny = {"tiny", 1, 1, 4, 4, 1, 3, 3, 0, 0, 1, 1};
    const ConvCase medium = {"medium", 1, 4, 16, 16, 4, 3, 3, 0, 0, 1, 1};

    bool ok = true;
    if (mode == "--medium") {
      ok = runCase(device, kernel, medium, runTimeoutMs);
    } else {
      ok = runCase(device, kernel, tiny, runTimeoutMs);
      if (ok && mode == "--all")
        ok = runCase(device, kernel, medium, runTimeoutMs);
    }

    alarm(0);
    return ok ? 0 : 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
