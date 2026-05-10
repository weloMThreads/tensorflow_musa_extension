#include <math.h>

#include <musa_runtime.h>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kILP = 1;
constexpr int kMaxBlocks = 4096;

__global__ void GeluGradFloatKernel(const float* __restrict__ x,
                                    const float* __restrict__ dy,
                                    float* __restrict__ dx, int n) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  constexpr float kInvSqrt2Pi = 0.39894228040143267794f;

  int i = (blockIdx.x * blockDim.x + threadIdx.x) * kILP;
  const int stride = blockDim.x * gridDim.x * kILP;
  for (; i < n; i += stride) {
#pragma unroll
    for (int j = 0; j < kILP; ++j) {
      const int idx = i + j;
      if (idx < n) {
        const float xv = x[idx];
        const float e = expf(-0.5f * xv * xv);
        const float cdf = 0.5f * (1.0f + erff(xv * kInvSqrt2));
        const float pdf_term = xv * e * kInvSqrt2Pi;
        dx[idx] = dy[idx] * (cdf + pdf_term);
      }
    }
  }
}

int ClampBlocks(int n) {
  int blocks = (n + kThreadsPerBlock * kILP - 1) /
               (kThreadsPerBlock * kILP);
  if (blocks < 1) return 1;
  return blocks > kMaxBlocks ? kMaxBlocks : blocks;
}

}  // namespace

extern "C" void LaunchMusaGeluGradFloat(const float* x, const float* dy,
                                        float* dx, int n,
                                        musaStream_t stream) {
  GeluGradFloatKernel<<<ClampBlocks(n), kThreadsPerBlock, 0, stream>>>(
      x, dy, dx, n);
}
