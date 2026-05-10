#include <math.h>
#include <stdint.h>

#include <musa_runtime.h>

namespace tensorflow {
namespace musa {
namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 8;
constexpr float kNegInf = -3.4028234663852886e38F;

__device__ __forceinline__ float WarpReduceMax(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

__device__ __forceinline__ float WarpReduceSum(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__global__ void SmallLastDimSoftmaxFloatKernel(const float* input,
                                               float* output,
                                               int64_t outer_size,
                                               int last_dim) {
  const int lane = threadIdx.x;
  const int row_in_block = threadIdx.y;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.y + row_in_block;
  if (row >= outer_size) return;

  const float* row_in = input + row * last_dim;
  float* row_out = output + row * last_dim;

  const int idx0 = lane;
  const int idx1 = lane + kWarpSize;

  const float x0 = (idx0 < last_dim) ? row_in[idx0] : kNegInf;
  const float x1 = (idx1 < last_dim) ? row_in[idx1] : kNegInf;

  const float row_max = WarpReduceMax(fmaxf(x0, x1));

  float e0 = 0.0f;
  float e1 = 0.0f;
  if (idx0 < last_dim) {
    e0 = __expf(x0 - row_max);
  }
  if (idx1 < last_dim) {
    e1 = __expf(x1 - row_max);
  }

  const float row_sum = WarpReduceSum(e0 + e1);
  const float inv_sum = 1.0f / row_sum;

  if (idx0 < last_dim) {
    row_out[idx0] = e0 * inv_sum;
  }
  if (idx1 < last_dim) {
    row_out[idx1] = e1 * inv_sum;
  }
}


__global__ void SmallLastDimSoftmaxFloatLe32Kernel(const float* input,
                                                   float* output,
                                                   int64_t outer_size,
                                                   int last_dim) {
  const int lane = threadIdx.x;
  const int row_in_block = threadIdx.y;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.y + row_in_block;
  if (row >= outer_size) return;

  const float* row_in = input + row * last_dim;
  float* row_out = output + row * last_dim;

  const float x = (lane < last_dim) ? row_in[lane] : kNegInf;
  const float row_max = WarpReduceMax(x);

  float e = 0.0f;
  if (lane < last_dim) {
    e = __expf(x - row_max);
  }

  const float row_sum = WarpReduceSum(e);
  const float inv_sum = 1.0f / row_sum;

  if (lane < last_dim) {
    row_out[lane] = e * inv_sum;
  }
}

__global__ void SmallLastDimCausalMaskedSoftmaxFloatKernel(
    const float* input, const float* scale_ptr, float* output,
    int64_t outer_size, int query_dim, int key_dim) {
  const int lane = threadIdx.x;
  const int row_in_block = threadIdx.y;
  const int query_idx = static_cast<int>(blockIdx.x) * blockDim.y + row_in_block;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * query_dim + query_idx;
  if (query_idx >= query_dim || row >= outer_size) return;

  const float scale = *scale_ptr;
  const int last_unmasked_key = key_dim - query_dim + query_idx;
  const float* row_in = input + row * key_dim;
  float* row_out = output + row * key_dim;

  const int idx0 = lane;
  const int idx1 = lane + kWarpSize;

  const bool valid0 = idx0 < key_dim && idx0 <= last_unmasked_key;
  const bool valid1 = idx1 < key_dim && idx1 <= last_unmasked_key;
  const float x0 = valid0 ? row_in[idx0] * scale : kNegInf;
  const float x1 = valid1 ? row_in[idx1] * scale : kNegInf;

  const float row_max = WarpReduceMax(fmaxf(x0, x1));

  float e0 = 0.0f;
  float e1 = 0.0f;
  if (valid0) e0 = __expf(x0 - row_max);
  if (valid1) e1 = __expf(x1 - row_max);

  const float row_sum = WarpReduceSum(e0 + e1);
  const float inv_sum = 1.0f / row_sum;

  if (idx0 < key_dim) row_out[idx0] = e0 * inv_sum;
  if (idx1 < key_dim) row_out[idx1] = e1 * inv_sum;
}

__global__ void SmallLastDimCausalMaskedSoftmaxFloatLe32Kernel(
    const float* input, const float* scale_ptr, float* output,
    int64_t outer_size, int query_dim, int key_dim) {
  const int lane = threadIdx.x;
  const int row_in_block = threadIdx.y;
  const int query_idx = static_cast<int>(blockIdx.x) * blockDim.y + row_in_block;
  const int64_t row = static_cast<int64_t>(blockIdx.y) * query_dim + query_idx;
  if (query_idx >= query_dim || row >= outer_size) return;

  const float scale = *scale_ptr;
  const int last_unmasked_key = key_dim - query_dim + query_idx;
  const float* row_in = input + row * key_dim;
  float* row_out = output + row * key_dim;

  const bool valid = lane < key_dim && lane <= last_unmasked_key;
  const float x = valid ? row_in[lane] * scale : kNegInf;
  const float row_max = WarpReduceMax(x);

  float e = 0.0f;
  if (valid) e = __expf(x - row_max);

  const float row_sum = WarpReduceSum(e);
  const float inv_sum = 1.0f / row_sum;

  if (lane < key_dim) row_out[lane] = e * inv_sum;
}

}  // namespace

extern "C" void LaunchMusaSmallLastDimSoftmaxFloat(const float* input,
                                                    float* output,
                                                    int64_t outer_size,
                                                    int last_dim,
                                                    musaStream_t stream) {
  if (outer_size <= 0 || last_dim <= 0 || last_dim > 64) return;
  const dim3 block(kWarpSize, kRowsPerBlock, 1);
  const dim3 grid((outer_size + kRowsPerBlock - 1) / kRowsPerBlock, 1, 1);
  if (last_dim <= kWarpSize) {
    constexpr int kRowsPerBlockLe32 = 16;
    const dim3 block_le32(kWarpSize, kRowsPerBlockLe32, 1);
    const dim3 grid_le32((outer_size + kRowsPerBlockLe32 - 1) /
                         kRowsPerBlockLe32, 1, 1);
    SmallLastDimSoftmaxFloatLe32Kernel<<<grid_le32, block_le32, 0, stream>>>(
        input, output, outer_size, last_dim);
  } else {
    SmallLastDimSoftmaxFloatKernel<<<grid, block, 0, stream>>>(
        input, output, outer_size, last_dim);
  }
}

extern "C" void LaunchMusaSmallLastDimCausalMaskedSoftmaxFloat(
    const float* input, const float* scale, float* output, int64_t outer_size,
    int query_dim, int key_dim, musaStream_t stream) {
  if (outer_size <= 0 || query_dim <= 0 || key_dim <= 0 ||
      query_dim > key_dim || key_dim > 64) {
    return;
  }
  if (key_dim <= kWarpSize) {
    constexpr int kRowsPerBlockLe32 = 16;
    const dim3 block(kWarpSize, kRowsPerBlockLe32, 1);
    const int64_t outer_groups = outer_size / query_dim;
    const dim3 grid((query_dim + kRowsPerBlockLe32 - 1) /
                        kRowsPerBlockLe32,
                    outer_groups, 1);
    SmallLastDimCausalMaskedSoftmaxFloatLe32Kernel<<<grid, block, 0, stream>>>(
        input, scale, output, outer_size, query_dim, key_dim);
  } else {
    const dim3 block(kWarpSize, kRowsPerBlock, 1);
    const int64_t outer_groups = outer_size / query_dim;
    const dim3 grid((query_dim + kRowsPerBlock - 1) / kRowsPerBlock,
                    outer_groups, 1);
    SmallLastDimCausalMaskedSoftmaxFloatKernel<<<grid, block, 0, stream>>>(
        input, scale, output, outer_size, query_dim, key_dim);
  }
}

}  // namespace musa
}  // namespace tensorflow
