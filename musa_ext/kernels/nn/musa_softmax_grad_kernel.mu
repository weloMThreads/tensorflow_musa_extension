#include <stdint.h>

#include <musa_runtime.h>

namespace tensorflow {
namespace musa {
namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 8;

__device__ __forceinline__ float WarpReduceSum(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__global__ void SmallLastDimSoftmaxGradFloatKernel(
    const float* __restrict__ softmax, const float* __restrict__ dy,
    float* __restrict__ dx, int64_t outer_size, int last_dim) {
  const int lane = threadIdx.x;
  const int row_in_block = threadIdx.y;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.y + row_in_block;
  if (row >= outer_size) return;

  const int idx0 = lane;
  const int idx1 = lane + kWarpSize;
  const int64_t base = row * last_dim;

  float local_sum = 0.0f;
  float y0 = 0.0f;
  float y1 = 0.0f;
  float dy0 = 0.0f;
  float dy1 = 0.0f;
  if (idx0 < last_dim) {
    y0 = softmax[base + idx0];
    dy0 = dy[base + idx0];
    local_sum += y0 * dy0;
  }
  if (idx1 < last_dim) {
    y1 = softmax[base + idx1];
    dy1 = dy[base + idx1];
    local_sum += y1 * dy1;
  }

  const float row_sum = WarpReduceSum(local_sum);
  if (idx0 < last_dim) {
    dx[base + idx0] = (dy0 - row_sum) * y0;
  }
  if (idx1 < last_dim) {
    dx[base + idx1] = (dy1 - row_sum) * y1;
  }
}

}  // namespace

extern "C" void LaunchMusaSmallLastDimSoftmaxGradFloat(
    const float* softmax, const float* dy, float* dx, int64_t outer_size,
    int last_dim, musaStream_t stream) {
  if (outer_size <= 0 || last_dim <= 0 || last_dim > 64) return;
  const dim3 block(kWarpSize, kRowsPerBlock, 1);
  const dim3 grid((outer_size + kRowsPerBlock - 1) / kRowsPerBlock, 1, 1);
  SmallLastDimSoftmaxGradFloatKernel<<<grid, block, 0, stream>>>(
      softmax, dy, dx, outer_size, last_dim);
}

}  // namespace musa
}  // namespace tensorflow
