#include <algorithm>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

using Clock = std::chrono::steady_clock;

void timeoutHandler(int) {
  const char message[] =
      "FAIL timeout: host_xrt_conv_test exceeded HOST_XRT_TIMEOUT_SEC\n";
  (void)!write(STDERR_FILENO, message, sizeof(message) - 1);
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

bool fileExists(const char *path) {
  struct stat status;
  return stat(path, &status) == 0 && S_ISREG(status.st_mode);
}

bool deviceNodeExists() {
  return access("/dev/dri/renderD128", R_OK | W_OK) == 0 ||
         access("/dev/zocl", R_OK | W_OK) == 0 ||
         access("/dev/xocl", R_OK | W_OK) == 0;
}

struct ConvCase {
  const char *name;
  int n;
  int c;
  int h;
  int w;
  int m;
  int kernel;
  int pad;
  int stride;
  bool hasBias;
};

double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void fillRandom(std::vector<float> &values, std::mt19937 &generator) {
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  for (float &value : values)
    value = distribution(generator);
}

void referenceConv(const ConvCase &test, const std::vector<float> &x,
    const std::vector<float> &weight, const std::vector<float> &bias,
    std::vector<float> &y, int ohSize, int owSize) {
  for (int n = 0; n < test.n; ++n)
    for (int m = 0; m < test.m; ++m)
      for (int oh = 0; oh < ohSize; ++oh)
        for (int ow = 0; ow < owSize; ++ow) {
          float sum = test.hasBias ? bias[m] : 0.0f;
          for (int c = 0; c < test.c; ++c)
            for (int kh = 0; kh < test.kernel; ++kh)
              for (int kw = 0; kw < test.kernel; ++kw) {
                const int ih = oh * test.stride + kh - test.pad;
                const int iw = ow * test.stride + kw - test.pad;
                if (ih < 0 || ih >= test.h || iw < 0 || iw >= test.w)
                  continue;
                const size_t xIndex =
                    ((size_t)n * test.c + c) * test.h * test.w +
                    (size_t)ih * test.w + iw;
                const size_t weightIndex =
                    (((size_t)m * test.c + c) * test.kernel + kh) *
                        test.kernel +
                    kw;
                sum += x[xIndex] * weight[weightIndex];
              }
          const size_t yIndex =
              ((size_t)n * test.m + m) * ohSize * owSize +
              (size_t)oh * owSize + ow;
          y[yIndex] = sum;
        }
}

bool runCase(xrt::device &device, xrt::kernel &kernel,
    const ConvCase &test, std::mt19937 &generator,
    unsigned int runTimeoutMs) {
  const int ohSize =
      (test.h + 2 * test.pad - test.kernel) / test.stride + 1;
  const int owSize =
      (test.w + 2 * test.pad - test.kernel) / test.stride + 1;
  const size_t xCount = (size_t)test.n * test.c * test.h * test.w;
  const size_t weightCount =
      (size_t)test.m * test.c * test.kernel * test.kernel;
  const size_t biasCount = test.hasBias ? (size_t)test.m : 1;
  const size_t yCount = (size_t)test.n * test.m * ohSize * owSize;
  const size_t xBytes = xCount * sizeof(float);
  const size_t weightBytes = weightCount * sizeof(float);
  const size_t biasBytes = biasCount * sizeof(float);
  const size_t yBytes = yCount * sizeof(float);

  std::vector<float> x(xCount);
  std::vector<float> weight(weightCount);
  std::vector<float> bias(test.m);
  std::vector<float> expected(yCount);
  std::vector<float> actual(yCount);
  fillRandom(x, generator);
  fillRandom(weight, generator);
  fillRandom(bias, generator);
  referenceConv(test, x, weight, bias, expected, ohSize, owSize);
  const float dummyBias = 0.0f;

  const auto allocationStart = Clock::now();
  xrt::bo xBo(device, xBytes, kernel.group_id(0));
  xrt::bo weightBo(device, weightBytes, kernel.group_id(1));
  xrt::bo biasBo(device, biasBytes, kernel.group_id(2));
  xrt::bo yBo(device, yBytes, kernel.group_id(3));
  const auto allocationEnd = Clock::now();

  const auto writeStart = Clock::now();
  xBo.write(x.data(), xBytes, 0);
  weightBo.write(weight.data(), weightBytes, 0);
  biasBo.write(test.hasBias ? bias.data() : &dummyBias, biasBytes, 0);
  const auto writeEnd = Clock::now();

  const auto h2dStart = Clock::now();
  xBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  weightBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  biasBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  const auto h2dEnd = Clock::now();

  const auto waitStart = Clock::now();
  ert_cmd_state state;
  if (test.kernel == 1) {
    auto run = kernel(xBo, weightBo, biasBo, yBo, test.n, test.c, test.h,
        test.w, test.m, test.hasBias);
    state = run.wait(runTimeoutMs);
    if (state == ERT_CMD_STATE_TIMEOUT)
      (void)run.abort();
  } else {
    auto run = kernel(xBo, weightBo, biasBo, yBo, test.n, test.c, test.h,
        test.w, test.m, ohSize, owSize, test.pad, test.pad, test.stride,
        test.stride, test.hasBias);
    state = run.wait(runTimeoutMs);
    if (state == ERT_CMD_STATE_TIMEOUT)
      (void)run.abort();
  }
  const auto waitEnd = Clock::now();
  if (state != ERT_CMD_STATE_COMPLETED) {
    std::fprintf(stderr, "FAIL %s kernel returned state=%s(%d)\n", test.name,
        stateName(state), (int)state);
    return false;
  }

  const auto d2hStart = Clock::now();
  yBo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const auto d2hEnd = Clock::now();
  const auto readStart = Clock::now();
  yBo.read(actual.data(), yBytes, 0);
  const auto readEnd = Clock::now();

  float maximumError = 0.0f;
  for (size_t i = 0; i < yCount; ++i) {
    const float error = std::fabs(actual[i] - expected[i]);
    maximumError = std::max(maximumError, error);
    const float tolerance = 2.0e-4f * std::max(1.0f, std::fabs(expected[i]));
    if (!std::isfinite(actual[i]) || error > tolerance) {
      std::fprintf(stderr,
          "FAIL %s at %zu: expected %.9g, got %.9g, error %.9g\n",
          test.name, i, expected[i], actual[i], error);
      return false;
    }
  }

  std::printf(
      "PASS %s max_abs_error=%.9g\n"
      "  bo_alloc=%8.3f ms  host_write=%8.3f ms  h2d_sync=%8.3f ms\n"
      "  run_wait=%8.3f ms  d2h_sync=%8.3f ms  host_read=%8.3f ms\n",
      test.name, maximumError, milliseconds(allocationStart, allocationEnd),
      milliseconds(writeStart, writeEnd), milliseconds(h2dStart, h2dEnd),
      milliseconds(waitStart, waitEnd), milliseconds(d2hStart, d2hEnd),
      milliseconds(readStart, readEnd));
  return true;
}

struct Int8ConvCase {
  const char *name;
  int n;
  int c;
  int h;
  int w;
  int m;
  int kernel;
  int pad;
  int stride;
  int xZeroPoint;
  int wZeroPoint;
  uint32_t multiplierBits;
  int outputZeroPoint;
};

float floatFromBits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

int8_t referenceRequantize(int32_t accumulator, int32_t bias,
    uint32_t multiplierBits, int outputZeroPoint) {
  // Preserve the production expression's two binary32 rounding points:
  // int64 -> float, followed by the float multiplication. This is exactly
  // nearbyintf((float)((int64)sum + bias) * multiplier) + output zero point.
  const int64_t combined = (int64_t)accumulator + (int64_t)bias;
  volatile float accumulatorFloat = (float)combined;
  volatile float multiplier = floatFromBits(multiplierBits);
  volatile float product = accumulatorFloat * multiplier;
  const int64_t shifted =
      (int64_t)::nearbyintf(product) + (int64_t)outputZeroPoint;
  if (shifted < -128)
    return (int8_t)-128;
  if (shifted > 127)
    return (int8_t)127;
  return (int8_t)shifted;
}

bool runInt8Case(xrt::device &device, xrt::kernel &kernel,
    const Int8ConvCase &test, std::mt19937 &generator,
    unsigned int runTimeoutMs) {
  const int ohSize =
      (test.h + 2 * test.pad - test.kernel) / test.stride + 1;
  const int owSize =
      (test.w + 2 * test.pad - test.kernel) / test.stride + 1;
  const size_t xCount = (size_t)test.n * test.c * test.h * test.w;
  const size_t weightCount =
      (size_t)test.m * test.c * test.kernel * test.kernel;
  const size_t outputCount =
      (size_t)test.n * test.m * ohSize * owSize;
  const size_t xBytes = xCount * sizeof(int8_t);
  const size_t weightBytes = weightCount * sizeof(int8_t);
  const size_t biasBytes = (size_t)test.m * sizeof(int32_t);
  const size_t outputBytes = outputCount * sizeof(int8_t);

  std::uniform_int_distribution<int> distribution(-128, 127);
  std::uniform_int_distribution<int32_t> biasDistribution(-10000, 10000);
  std::vector<int8_t> x(xCount);
  std::vector<int8_t> weight(weightCount);
  std::vector<int32_t> bias(test.m);
  std::vector<int8_t> expected(outputCount);
  std::vector<int8_t> actual(outputCount, 0);
  for (int8_t &value : x)
    value = (int8_t)distribution(generator);
  for (int8_t &value : weight)
    value = (int8_t)distribution(generator);
  for (int32_t &value : bias)
    value = biasDistribution(generator);

  for (int n = 0; n < test.n; ++n)
    for (int m = 0; m < test.m; ++m)
      for (int oh = 0; oh < ohSize; ++oh)
        for (int ow = 0; ow < owSize; ++ow) {
          int32_t sum = 0;
          for (int c = 0; c < test.c; ++c)
            for (int kh = 0; kh < test.kernel; ++kh)
              for (int kw = 0; kw < test.kernel; ++kw) {
                const int ih = oh * test.stride + kh - test.pad;
                const int iw = ow * test.stride + kw - test.pad;
                int32_t input = test.xZeroPoint;
                if (ih >= 0 && ih < test.h && iw >= 0 && iw < test.w) {
                  const size_t xIndex =
                      ((size_t)n * test.c + c) * test.h * test.w +
                      (size_t)ih * test.w + iw;
                  input = x[xIndex];
                }
                const size_t weightIndex =
                    (((size_t)m * test.c + c) * test.kernel + kh) *
                        test.kernel +
                    kw;
                sum += (input - test.xZeroPoint) *
                       ((int32_t)weight[weightIndex] - test.wZeroPoint);
              }
          const size_t outputIndex =
              ((size_t)n * test.m + m) * ohSize * owSize +
              (size_t)oh * owSize + ow;
          expected[outputIndex] = referenceRequantize(sum, bias[m],
              test.multiplierBits, test.outputZeroPoint);
        }

  const auto allocationStart = Clock::now();
  xrt::bo xBo(device, xBytes, kernel.group_id(0));
  xrt::bo weightBo(device, weightBytes, kernel.group_id(1));
  xrt::bo biasBo(device, biasBytes, kernel.group_id(2));
  xrt::bo outputBo(device, outputBytes, kernel.group_id(3));
  const auto allocationEnd = Clock::now();

  const auto writeStart = Clock::now();
  xBo.write(x.data(), xBytes, 0);
  weightBo.write(weight.data(), weightBytes, 0);
  biasBo.write(bias.data(), biasBytes, 0);
  const auto writeEnd = Clock::now();

  const auto h2dStart = Clock::now();
  xBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  weightBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  biasBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  const auto h2dEnd = Clock::now();

  const auto waitStart = Clock::now();
  ert_cmd_state state;
  if (test.kernel == 1) {
    auto run = kernel(xBo, weightBo, biasBo, outputBo, test.n, test.c, test.h,
        test.w, test.m, test.xZeroPoint, test.wZeroPoint,
        test.multiplierBits, test.outputZeroPoint);
    state = run.wait(runTimeoutMs);
    if (state == ERT_CMD_STATE_TIMEOUT)
      (void)run.abort();
  } else {
    auto run = kernel(xBo, weightBo, biasBo, outputBo, test.n, test.c, test.h,
        test.w, test.m, ohSize, owSize, test.pad, test.pad, test.stride,
        test.stride, test.xZeroPoint, test.wZeroPoint, test.multiplierBits,
        test.outputZeroPoint);
    state = run.wait(runTimeoutMs);
    if (state == ERT_CMD_STATE_TIMEOUT)
      (void)run.abort();
  }
  const auto waitEnd = Clock::now();
  if (state != ERT_CMD_STATE_COMPLETED) {
    std::fprintf(stderr, "FAIL %s returned state=%s(%d)\n", test.name,
        stateName(state), (int)state);
    return false;
  }

  const auto d2hStart = Clock::now();
  outputBo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const auto d2hEnd = Clock::now();
  const auto readStart = Clock::now();
  outputBo.read(actual.data(), outputBytes, 0);
  const auto readEnd = Clock::now();

  for (size_t i = 0; i < outputCount; ++i) {
    if (actual[i] != expected[i]) {
      std::fprintf(stderr, "FAIL %s at %zu: expected %d, got %d\n",
          test.name, i, (int)expected[i], (int)actual[i]);
      return false;
    }
  }

  std::printf(
      "PASS %s exact_int8_outputs=%zu multiplier_bits=0x%08x "
      "output_zero_point=%d\n"
      "  bo_alloc=%8.3f ms  host_write=%8.3f ms  h2d_sync=%8.3f ms\n"
      "  run_wait=%8.3f ms  d2h_sync=%8.3f ms  host_read=%8.3f ms\n",
      test.name, outputCount, test.multiplierBits, test.outputZeroPoint,
      milliseconds(allocationStart, allocationEnd),
      milliseconds(writeStart, writeEnd), milliseconds(h2dStart, h2dEnd),
      milliseconds(waitStart, waitEnd), milliseconds(d2hStart, d2hEnd),
      milliseconds(readStart, readEnd));
  return true;
}

} // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);

  if (argc < 2 || argc > 3) {
    std::fprintf(stderr,
        "usage: %s <xclbin> "
        "[--probe-only|--1x1-only|--3x3-only|--int8-only|--int8-all|"
        "--int8-1x1-only|--int8-3x3-only|--int8-6x6-only]\n",
        argv[0]);
    return 2;
  }

  const bool probeOnly =
      argc == 3 && std::strcmp(argv[2], "--probe-only") == 0;
  const bool run1x1 = argc == 2 || std::strcmp(argv[2], "--1x1-only") == 0;
  const bool run3x3 = argc == 2 || std::strcmp(argv[2], "--3x3-only") == 0;
  const bool runInt8_1x1 = argc == 2 ||
      std::strcmp(argv[2], "--int8-only") == 0 ||
      std::strcmp(argv[2], "--int8-all") == 0 ||
      std::strcmp(argv[2], "--int8-1x1-only") == 0;
  const bool runInt8_3x3 = argc == 2 ||
      std::strcmp(argv[2], "--int8-only") == 0 ||
      std::strcmp(argv[2], "--int8-all") == 0 ||
      std::strcmp(argv[2], "--int8-3x3-only") == 0;
  const bool runInt8_6x6 = argc == 2 ||
      std::strcmp(argv[2], "--int8-all") == 0 ||
      std::strcmp(argv[2], "--int8-6x6-only") == 0;
  if (!probeOnly && !run1x1 && !run3x3 && !runInt8_1x1 &&
      !runInt8_3x3 && !runInt8_6x6) {
    std::fprintf(stderr, "error: unknown option: %s\n", argv[2]);
    return 2;
  }

  const unsigned int timeoutSec = envUInt("HOST_XRT_TIMEOUT_SEC", 30);
  const unsigned int runTimeoutMs = envUInt("HOST_XRT_RUN_TIMEOUT_MS", 5000);
  std::signal(SIGALRM, timeoutHandler);
  if (timeoutSec > 0)
    alarm(timeoutSec);

  if (!fileExists(argv[1])) {
    std::fprintf(
        stderr, "FAIL preflight: xclbin does not exist: %s\n", argv[1]);
    return 2;
  }
  if (!deviceNodeExists()) {
    std::fprintf(stderr,
        "FAIL preflight: no usable XRT device node found "
        "(/dev/dri/renderD128, /dev/zocl, or /dev/xocl)\n");
    return 2;
  }

  try {
    xrt::device device(0);
    const xrt::uuid uuid = device.load_xclbin(argv[1]);
    if (probeOnly) {
      xrt::kernel conv1x1(device, uuid, "conv1x1_kernel");
      xrt::kernel conv3x3(device, uuid, "conv3x3_kernel");
      xrt::kernel conv1x1Int8(device, uuid, "conv1x1_i8_kernel");
      xrt::kernel conv3x3Int8(device, uuid, "conv3x3_i8_kernel");
      xrt::kernel conv6x6StemInt8(
          device, uuid, "conv6x6_stem_i8_kernel");
      std::printf("PASS loaded xclbin and opened all kernel handles\n");
      alarm(0);
      return 0;
    }
    std::mt19937 generator(0x498);
    bool passed = true;
    if (run1x1) {
      xrt::kernel kernel(device, uuid, "conv1x1_kernel");
      const ConvCase test = {
          "conv1x1_kernel", 1, 5, 4, 7, 17, 1, 0, 1, true};
      passed =
          runCase(device, kernel, test, generator, runTimeoutMs) && passed;
    }
    if (run3x3) {
      xrt::kernel kernel(device, uuid, "conv3x3_kernel");
      const ConvCase test = {
          "conv3x3_kernel", 1, 5, 7, 6, 17, 3, 1, 2, true};
      passed =
          runCase(device, kernel, test, generator, runTimeoutMs) && passed;
    }
    if (runInt8_1x1) {
      xrt::kernel kernel(device, uuid, "conv1x1_i8_kernel");
      const Int8ConvCase test = {
          "conv1x1_i8_kernel", 1, 4, 4, 7, 17, 1, 0, 1, -7, 11,
          0x3b000000U, -9};
      passed = runInt8Case(
                   device, kernel, test, generator, runTimeoutMs) &&
          passed;
    }
    if (runInt8_3x3) {
      xrt::kernel kernel(device, uuid, "conv3x3_i8_kernel");
      const Int8ConvCase test = {
          "conv3x3_i8_kernel", 1, 4, 7, 8, 17, 3, 1, 2, -7, 11,
          0x39800000U, 13};
      passed = runInt8Case(
                   device, kernel, test, generator, runTimeoutMs) &&
          passed;
    }
    if (runInt8_6x6) {
      xrt::kernel kernel(device, uuid, "conv6x6_stem_i8_kernel");
      const Int8ConvCase test = {
          "conv6x6_stem_i8_kernel", 1, 3, 8, 8, 7, 6, 2, 2, -7, 11,
          0x39000000U, -3};
      passed = runInt8Case(
                   device, kernel, test, generator, runTimeoutMs) &&
          passed;
    }
    alarm(0);
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "XRT test failed: %s\n", error.what());
    return 1;
  }
}
