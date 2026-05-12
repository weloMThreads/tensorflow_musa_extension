#include <musa_runtime.h>

#include <cstdint>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-pragmas"
#include "tensorflow/core/framework/bfloat16.h"
#pragma GCC diagnostic pop

__device__ __forceinline__ float LoadBFloat16(const tensorflow::bfloat16* p) {
  float res = 0.0f;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(p);
  uint32_t* f_ptr = reinterpret_cast<uint32_t*>(&res);
  *f_ptr = static_cast<uint32_t>(*b_ptr) << 16;
  return res;
}

__device__ __forceinline__ void StoreBFloat16(tensorflow::bfloat16* p,
                                              float v) {
  const uint32_t* f_ptr = reinterpret_cast<const uint32_t*>(&v);
  uint16_t* b_ptr = reinterpret_cast<uint16_t*>(p);
  *b_ptr = static_cast<uint16_t>(*f_ptr >> 16);
}

__global__ void ScaledMaskedMulFloatKernel(const float* x, const bool* mask,
                                           float* y, int64_t n, float scale) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    y[i] = mask[i] ? x[i] * scale : 0.0f;
  }
}

__global__ void ScaledMaskedMulBFloat16Kernel(const tensorflow::bfloat16* x,
                                              const bool* mask,
                                              tensorflow::bfloat16* y,
                                              int64_t n, float scale) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    StoreBFloat16(&y[i], mask[i] ? LoadBFloat16(&x[i]) * scale : 0.0f);
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

extern "C" void LaunchMusaScaledMaskedMulBFloat16(
    const tensorflow::bfloat16* x, const bool* mask, tensorflow::bfloat16* y,
    int64_t n, float scale, musaStream_t stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  if (blocks > 4096) blocks = 4096;
  ScaledMaskedMulBFloat16Kernel<<<blocks, kThreads, 0, stream>>>(x, mask, y, n,
                                                                 scale);
}
