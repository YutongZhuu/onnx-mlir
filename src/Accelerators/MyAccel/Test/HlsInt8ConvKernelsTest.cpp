#include "Conv1x1Int8Kernel.h"
#include "Conv3x3Int8Kernel.h"
#include "Int8Requantize.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

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

int64_t referenceFloatMultiplyRoundEven(
    int64_t accumulator, uint32_t multiplierBits) {
  // Volatile temporaries force the two binary32 rounding points used by the
  // production expression: int64 -> float, then float multiplication.
  volatile float accumulatorFloat = (float)accumulator;
  volatile float multiplier = floatFromBits(multiplierBits);
  volatile float product = accumulatorFloat * multiplier;
  return (int64_t)::nearbyintf(product);
}

int8_t referenceRequantize(int32_t accumulator, int32_t bias,
    uint32_t multiplierBits, int outputZeroPoint) {
  const int64_t combined = (int64_t)accumulator + (int64_t)bias;
  const int64_t scaled =
      referenceFloatMultiplyRoundEven(combined, multiplierBits);
  const int64_t shifted = scaled + (int64_t)outputZeroPoint;
  if (shifted < -128)
    return (int8_t)-128;
  if (shifted > 127)
    return (int8_t)127;
  return (int8_t)shifted;
}

bool testDirectedRequantization() {
  struct RequantCase {
    const char *name;
    int32_t accumulator;
    int32_t bias;
    uint32_t multiplierBits;
    int outputZeroPoint;
    int expected;
  };

  const RequantCase cases[] = {
      {"positive half to even zero", 1, 0, 0x3f000000U, 0, 0},
      {"positive one-and-half to even", 3, 0, 0x3f000000U, 0, 2},
      {"positive two-and-half to even", 5, 0, 0x3f000000U, 0, 2},
      {"negative half to even zero", -1, 0, 0x3f000000U, 0, 0},
      {"negative one-and-half to even", -3, 0, 0x3f000000U, 0, -2},
      {"negative two-and-half to even", -5, 0, 0x3f000000U, 0, -2},
      {"bias before requantization", 0, 3, 0x3f000000U, 0, 2},
      {"nonzero output zero point", 4, 0, 0x3e800000U, 17, 18},
      {"positive saturation", 1000, 0, 0x3f000000U, 0, 127},
      {"negative saturation", -1000, 0, 0x3f000000U, 0, -128},
      {"multiplier below one", 8, 0, 0x3e800000U, 0, 2},
      {"multiplier above one", 2, 0, 0x3fc00000U, 0, 3},
      {"maximum positive bias", 0, INT32_MAX, 0x3f000000U, 0, 127},
      {"minimum negative bias", 0, INT32_MIN, 0x3f000000U, 0, -128},
      // A single-rounding Q31 implementation returns +1 here. The host first
      // rounds 16,777,217 to binary32 16,777,216, making the product exactly
      // 0.5, which ties to even zero.
      {"int64-to-float counterexample positive", 16777217, 0,
          0x33000000U, 0, 0},
      // Exercise a combined accumulator below INT32_MIN. Direct fixed-point
      // rounding returns -1, while binary32 conversion makes this -2^31 and
      // the product -0.5, which ties to even zero.
      {"int64-to-float counterexample negative", INT32_MIN, -1,
          0x2f800000U, 0, 0},
  };

  for (const RequantCase &test : cases) {
    const int actual = myaccel_int8::requantize(test.accumulator, test.bias,
        test.multiplierBits, test.outputZeroPoint);
    if (actual != test.expected) {
      std::fprintf(stderr, "FAIL %-42s expected %d, got %d\n", test.name,
          test.expected, actual);
      return false;
    }
  }
  std::printf("PASS %-42s cases=%zu\n", "directed requantization",
      sizeof(cases) / sizeof(cases[0]));
  return true;
}

bool testFloatEmulationDifferential() {
  const uint32_t directedMultipliers[] = {
      0x2f800000U, // 2^-32
      0x33000000U, // 2^-25
      0x3dcccccdU, // nearest binary32 to 0.1
      0x3e800000U, // 0.25
      0x3f000000U, // 0.5
      0x3f000001U, // next binary32 above 0.5
      0x3f400000U, // 0.75
      0x3f800000U, // 1.0
      0x3fc00000U, // 1.5
  };
  const int64_t directedAccumulators[] = {
      0,
      1,
      -1,
      3,
      -3,
      16777215,
      16777216,
      16777217,
      -16777217,
      (int64_t)INT32_MAX + INT32_MAX,
      (int64_t)INT32_MIN + INT32_MIN,
      (int64_t)INT32_MIN - 1,
  };

  size_t comparisons = 0;
  for (uint32_t multiplierBits : directedMultipliers)
    for (int64_t accumulator : directedAccumulators) {
      const int64_t expected =
          referenceFloatMultiplyRoundEven(accumulator, multiplierBits);
      const int64_t actual = myaccel_int8::emulateFloatMultiplyRoundEven(
          accumulator, multiplierBits);
      ++comparisons;
      if (actual != expected) {
        std::fprintf(stderr,
            "FAIL float emulation acc=%lld multiplier=0x%08x expected=%lld "
            "got=%lld\n",
            (long long)accumulator, multiplierBits, (long long)expected,
            (long long)actual);
        return false;
      }
    }

  std::mt19937 generator(0x493);
  for (int iteration = 0; iteration < 200000; ++iteration) {
    const int32_t accumulator = (int32_t)generator();
    const int32_t bias = (int32_t)generator();
    const int64_t combined = (int64_t)accumulator + bias;
    // Cover positive normal multipliers from about 2^-47 through 2^3 with a
    // random 23-bit fraction. This includes the full range used by the model.
    const uint32_t exponentBits = 80U + generator() % 51U;
    const uint32_t multiplierBits =
        (exponentBits << 23) | (generator() & 0x7fffffU);
    const int64_t expected =
        referenceFloatMultiplyRoundEven(combined, multiplierBits);
    const int64_t actual = myaccel_int8::emulateFloatMultiplyRoundEven(
        combined, multiplierBits);
    ++comparisons;
    if (actual != expected) {
      std::fprintf(stderr,
          "FAIL random float emulation iteration=%d acc=%lld "
          "multiplier=0x%08x expected=%lld got=%lld\n",
          iteration, (long long)combined, multiplierBits,
          (long long)expected, (long long)actual);
      return false;
    }
  }

  std::printf("PASS %-42s comparisons=%zu\n", "binary32 differential",
      comparisons);
  return true;
}

void referenceConv(const ConvCase &test, const std::vector<int8_t> &x,
    const std::vector<int8_t> &weight, const std::vector<int32_t> &bias,
    std::vector<int8_t> &output, int ohSize, int owSize) {
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
          output[outputIndex] = referenceRequantize(sum, bias[m],
              test.multiplierBits, test.outputZeroPoint);
        }
}

bool runCase(const ConvCase &test, std::mt19937 &generator) {
  const int ohSize =
      (test.h + 2 * test.pad - test.kernel) / test.stride + 1;
  const int owSize =
      (test.w + 2 * test.pad - test.kernel) / test.stride + 1;
  const size_t xCount = (size_t)test.n * test.c * test.h * test.w;
  const size_t weightCount =
      (size_t)test.m * test.c * test.kernel * test.kernel;
  const size_t outputCount =
      (size_t)test.n * test.m * ohSize * owSize;

  std::uniform_int_distribution<int> valueDistribution(-128, 127);
  std::uniform_int_distribution<int32_t> biasDistribution(-10000, 10000);
  std::vector<int8_t> x(xCount);
  std::vector<int8_t> weight(weightCount);
  std::vector<int32_t> bias(test.m);
  for (int8_t &value : x)
    value = (int8_t)valueDistribution(generator);
  for (int8_t &value : weight)
    value = (int8_t)valueDistribution(generator);
  for (int32_t &value : bias)
    value = biasDistribution(generator);

  std::vector<int8_t> expected(outputCount);
  std::vector<int8_t> actual(outputCount, 0);
  referenceConv(test, x, weight, bias, expected, ohSize, owSize);
  if (test.kernel == 1) {
    conv1x1_i8_kernel(x.data(), weight.data(), bias.data(), actual.data(),
        test.n, test.c, test.h, test.w, test.m, test.xZeroPoint,
        test.wZeroPoint, test.multiplierBits, test.outputZeroPoint);
  } else {
    conv3x3_i8_kernel(x.data(), weight.data(), bias.data(), actual.data(),
        test.n, test.c, test.h, test.w, test.m, ohSize, owSize, test.pad,
        test.pad, test.stride, test.stride, test.xZeroPoint, test.wZeroPoint,
        test.multiplierBits, test.outputZeroPoint);
  }

  for (size_t i = 0; i < outputCount; ++i) {
    if (actual[i] != expected[i]) {
      std::fprintf(stderr, "FAIL %-30s at %zu: expected %d, got %d\n",
          test.name, i, (int)expected[i], (int)actual[i]);
      return false;
    }
  }
  std::printf("PASS %-30s outputs=%zu\n", test.name, outputCount);
  return true;
}

} // namespace

int main() {
  const ConvCase tests[] = {
      {"INT8 1x1 nonzero zero-points", 2, 5, 3, 4, 7, 1, 0, 1, 3, -5,
          0x3dcccccdU, 11},
      {"INT8 1x1 channel/output tails", 1, 17, 4, 5, 19, 1, 0, 1, -7,
          11, 0x3b800000U, -9},
      {"INT8 1x1 multiplier above one", 1, 9, 2, 17, 17, 1, 0, 1, 0, 0,
          0x3fc00000U, 3},
      {"INT8 3x3 stride1 edge tiles", 1, 3, 7, 6, 5, 3, 1, 1, 5, -9,
          0x3a800000U, -7},
      {"INT8 3x3 stride2 tails", 2, 5, 8, 9, 17, 3, 1, 2, -3, 6,
          0x39800000U, 13},
      {"INT8 3x3 max channel tail", 1, 128, 4, 4, 17, 3, 1, 1, 0, 0,
          0x38800000U, 0},
  };

  std::mt19937 generator(0x498);
  bool passed = testDirectedRequantization();
  passed = testFloatEmulationDifferential() && passed;
  for (const ConvCase &test : tests)
    passed = runCase(test, generator) && passed;
  return passed ? 0 : 1;
}
