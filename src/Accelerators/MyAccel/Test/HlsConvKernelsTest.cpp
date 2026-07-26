#include "Conv1x1Kernel.h"
#include "Conv3x3Kernel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
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
  bool hasBias;
};

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

bool runCase(const ConvCase &test, std::mt19937 &generator) {
  const int ohSize =
      (test.h + 2 * test.pad - test.kernel) / test.stride + 1;
  const int owSize =
      (test.w + 2 * test.pad - test.kernel) / test.stride + 1;
  const size_t xCount = (size_t)test.n * test.c * test.h * test.w;
  const size_t weightCount =
      (size_t)test.m * test.c * test.kernel * test.kernel;
  const size_t yCount = (size_t)test.n * test.m * ohSize * owSize;

  std::vector<float> x(xCount);
  std::vector<float> weight(weightCount);
  std::vector<float> bias(test.m);
  std::vector<float> expected(yCount);
  std::vector<float> actual(yCount, 1234.0f);
  fillRandom(x, generator);
  fillRandom(weight, generator);
  fillRandom(bias, generator);

  referenceConv(test, x, weight, bias, expected, ohSize, owSize);
  if (test.kernel == 1) {
    conv1x1_kernel(x.data(), weight.data(), bias.data(), actual.data(), test.n,
        test.c, test.h, test.w, test.m, test.hasBias);
  } else {
    conv3x3_kernel(x.data(), weight.data(), bias.data(), actual.data(), test.n,
        test.c, test.h, test.w, test.m, ohSize, owSize, test.pad, test.pad,
        test.stride, test.stride, test.hasBias);
  }

  float maximumError = 0.0f;
  for (size_t i = 0; i < yCount; ++i) {
    const float error = std::fabs(actual[i] - expected[i]);
    maximumError = std::max(maximumError, error);
    const float tolerance = 2.0e-4f * std::max(1.0f, std::fabs(expected[i]));
    if (!std::isfinite(actual[i]) || error > tolerance) {
      std::fprintf(stderr,
          "FAIL %-24s at %zu: expected %.9g, got %.9g, error %.9g\n",
          test.name, i, expected[i], actual[i], error);
      return false;
    }
  }

  std::printf("PASS %-24s max_abs_error=%.9g\n", test.name, maximumError);
  return true;
}

} // namespace

int main() {
  const ConvCase tests[] = {
      {"1x1 channel/output tails", 1, 3, 3, 5, 5, 1, 0, 1, true},
      {"1x1 batch and block tail", 2, 5, 2, 3, 17, 1, 0, 1, false},
      {"1x1 max channel tail", 1, 511, 1, 2, 17, 1, 0, 1, true},
      {"3x3 stride1 edge tiles", 1, 3, 5, 7, 5, 3, 1, 1, true},
      {"3x3 stride2 tails", 1, 5, 7, 6, 17, 3, 1, 2, false},
      {"3x3 batch no padding", 2, 2, 4, 5, 3, 3, 0, 1, true},
      {"3x3 max channel tail", 1, 127, 3, 4, 17, 3, 1, 1, true},
  };

  std::mt19937 generator(0x498);
  bool passed = true;
  for (const ConvCase &test : tests)
    passed = runCase(test, generator) && passed;
  return passed ? 0 : 1;
}
