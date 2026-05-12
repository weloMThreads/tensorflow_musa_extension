#include <musa_bf16.h>
#include <musa_runtime.h>

#include <stdint.h>

extern "C" {

__global__ void ResourceApplyGradientDescentFloatBFloat16Kernel(
    float* __restrict__ var, const __mt_bfloat16* __restrict__ grad,
    float alpha, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    var[i] -= alpha * __bfloat162float(grad[i]);
  }
}

void LaunchResourceApplyGradientDescentFloatBFloat16(
    float* var, const void* grad, float alpha, int64_t n, musaStream_t stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  ResourceApplyGradientDescentFloatBFloat16Kernel<<<blocks, kThreads, 0,
                                                    stream>>>(
      var, reinterpret_cast<const __mt_bfloat16*>(grad), alpha, n);
}

}  // extern "C"
