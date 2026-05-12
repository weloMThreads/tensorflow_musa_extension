#include <stdint.h>

#include <musa_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

__global__ void ResourceApplyGradientDescentFloatKernel(
    float* __restrict__ var, const float* __restrict__ grad, float lr,
    int64_t n) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    var[i] -= lr * grad[i];
  }
}

}  // namespace

extern "C" void LaunchResourceApplyGradientDescentFloat(
    float* var, const float* grad, float lr, int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  if (blocks < 1) {
    blocks = 1;
  } else if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }
  ResourceApplyGradientDescentFloatKernel<<<blocks, kThreads, 0, stream>>>(
      var, grad, lr, n);
}
