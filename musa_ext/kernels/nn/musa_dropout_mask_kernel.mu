#include <musa_runtime.h>

#include <cstdint>

__global__ void ScaledMaskedMulFloatKernel(const float* x, const bool* mask,
                                           float* y, int64_t n, float scale) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    y[i] = mask[i] ? x[i] * scale : 0.0f;
  }
}

extern "C" void LaunchMusaScaledMaskedMulFloat(const float* x,
                                                const bool* mask, float* y,
                                                int64_t n, float scale,
                                                musaStream_t stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  if (blocks > 4096) blocks = 4096;
  ScaledMaskedMulFloatKernel<<<blocks, kThreads, 0, stream>>>(x, mask, y, n,
                                                              scale);
}
