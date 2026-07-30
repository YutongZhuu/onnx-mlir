#include "Conv1x1Int8Kernel.h"
#include "Conv3x3Int8Kernel.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
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
};

void referenceConv(const ConvCase &test, const std::vector<int8_t> &x,
    const std::vector<int8_t> &weight, std::vector<int32_t> &accumulator,
    int ohSize, int owSize) {
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
          accumulator[outputIndex] = sum;
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

  std::uniform_int_distribution<int> distribution(-128, 127);
  std::vector<int8_t> x(xCount);
  std::vector<int8_t> weight(weightCount);
  for (int8_t &value : x)
    value = (int8_t)distribution(generator);
  for (int8_t &value : weight)
    value = (int8_t)distribution(generator);

  std::vector<int32_t> expected(outputCount);
  std::vector<int32_t> actual(outputCount, INT32_MIN);
  referenceConv(test, x, weight, expected, ohSize, owSize);
  if (test.kernel == 1) {
    conv1x1_i8_kernel(x.data(), weight.data(), actual.data(), test.n, test.c,
        test.h, test.w, test.m, test.xZeroPoint, test.wZeroPoint);
  } else {
    conv3x3_i8_kernel(x.data(), weight.data(), actual.data(), test.n, test.c,
        test.h, test.w, test.m, ohSize, owSize, test.pad, test.pad,
        test.stride, test.stride, test.xZeroPoint, test.wZeroPoint);
  }

  for (size_t i = 0; i < outputCount; ++i) {
    if (actual[i] != expected[i]) {
      std::fprintf(stderr, "FAIL %-30s at %zu: expected %d, got %d\n",
          test.name, i, expected[i], actual[i]);
      return false;
    }
  }
  std::printf("PASS %-30s outputs=%zu\n", test.name, outputCount);
  return true;
}

} // namespace

int main() {
  const ConvCase tests[] = {
      {"INT8 1x1 nonzero zero-points", 2, 5, 3, 4, 7, 1, 0, 1, 3, -5},
      {"INT8 1x1 channel/output tails", 1, 17, 4, 5, 19, 1, 0, 1, -7, 11},
      {"INT8 3x3 stride1 edge tiles", 1, 3, 7, 6, 5, 3, 1, 1, 5, -9},
      {"INT8 3x3 stride2 tails", 2, 5, 8, 9, 17, 3, 1, 2, -3, 6},
      {"INT8 3x3 max channel tail", 1, 128, 4, 4, 17, 3, 1, 1, 0, 0},
  };

  std::mt19937 generator(0x498);
  bool passed = true;
  for (const ConvCase &test : tests)
    passed = runCase(test, generator) && passed;
  return passed ? 0 : 1;
}
