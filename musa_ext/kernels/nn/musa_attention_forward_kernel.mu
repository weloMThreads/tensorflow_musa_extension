#include <math.h>
#include <stdint.h>

#include <musa_runtime.h>

namespace tensorflow {
namespace musa {
namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 4;
constexpr float kNegInf = -3.4028234663852886e38F;

__device__ __forceinline__ float WarpReduceMax(float value) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask));
  }
  return value;
}

__device__ __forceinline__ float WarpReduceSum(float value) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, mask);
  }
  return value;
}

__global__ void CausalAttentionForwardDh8FloatKernel(
    const float* __restrict__ query, const float* __restrict__ key,
    const float* __restrict__ value, const float* __restrict__ scale_ptr,
    float* __restrict__ softmax, float* __restrict__ output, int64_t groups,
    int query_dim, int key_dim) {
  const int lane = threadIdx.x;
  const int row_in_block = threadIdx.y;
  const int query_idx = static_cast<int>(blockIdx.x) * blockDim.y + row_in_block;
  const int64_t group = static_cast<int64_t>(blockIdx.y);
  if (group >= groups || query_idx >= query_dim) return;

  constexpr int kHeadDim = 8;
  const int key_idx0 = lane;
  const int key_idx1 = lane + kWarpSize;
  const int last_unmasked_key = key_dim - query_dim + query_idx;
  const float scale = *scale_ptr;

  const int64_t query_base =
      (group * static_cast<int64_t>(query_dim) + query_idx) * kHeadDim;
  const int64_t key_base = group * static_cast<int64_t>(key_dim) * kHeadDim;
  const int64_t softmax_base =
      (group * static_cast<int64_t>(query_dim) + query_idx) * key_dim;
  const int64_t output_base =
      (group * static_cast<int64_t>(query_dim) + query_idx) * kHeadDim;

  const float q0 = query[query_base + 0];
  const float q1 = query[query_base + 1];
  const float q2 = query[query_base + 2];
  const float q3 = query[query_base + 3];
  const float q4 = query[query_base + 4];
  const float q5 = query[query_base + 5];
  const float q6 = query[query_base + 6];
  const float q7 = query[query_base + 7];

  auto dot_key = [&](int key_idx) {
    const float* key_ptr = key + key_base + static_cast<int64_t>(key_idx) * kHeadDim;
    return (q0 * key_ptr[0] + q1 * key_ptr[1] +
            q2 * key_ptr[2] + q3 * key_ptr[3] +
            q4 * key_ptr[4] + q5 * key_ptr[5] +
            q6 * key_ptr[6] + q7 * key_ptr[7]) * scale;
  };

  const bool valid0 = key_idx0 < key_dim && key_idx0 <= last_unmasked_key;
  const bool valid1 = key_idx1 < key_dim && key_idx1 <= last_unmasked_key;
  const float x0 = valid0 ? dot_key(key_idx0) : kNegInf;
  const float x1 = valid1 ? dot_key(key_idx1) : kNegInf;

  const float row_max = WarpReduceMax(fmaxf(x0, x1));
  const float e0 = valid0 ? __expf(x0 - row_max) : 0.0f;
  const float e1 = valid1 ? __expf(x1 - row_max) : 0.0f;
  const float row_sum = WarpReduceSum(e0 + e1);
  const float inv_sum = 1.0f / row_sum;
  const float p0 = e0 * inv_sum;
  const float p1 = e1 * inv_sum;

  if (key_idx0 < key_dim) softmax[softmax_base + key_idx0] = p0;
  if (key_idx1 < key_dim) softmax[softmax_base + key_idx1] = p1;

  const float* value_base = value + group * static_cast<int64_t>(key_dim) * kHeadDim;
  float partial0 = 0.0f;
  float partial1 = 0.0f;
  float partial2 = 0.0f;
  float partial3 = 0.0f;
  float partial4 = 0.0f;
  float partial5 = 0.0f;
  float partial6 = 0.0f;
  float partial7 = 0.0f;
  if (key_idx0 < key_dim) {
    const float* v = value_base + static_cast<int64_t>(key_idx0) * kHeadDim;
    partial0 += p0 * v[0];
    partial1 += p0 * v[1];
    partial2 += p0 * v[2];
    partial3 += p0 * v[3];
    partial4 += p0 * v[4];
    partial5 += p0 * v[5];
    partial6 += p0 * v[6];
    partial7 += p0 * v[7];
  }
  if (key_idx1 < key_dim) {
    const float* v = value_base + static_cast<int64_t>(key_idx1) * kHeadDim;
    partial0 += p1 * v[0];
    partial1 += p1 * v[1];
    partial2 += p1 * v[2];
    partial3 += p1 * v[3];
    partial4 += p1 * v[4];
    partial5 += p1 * v[5];
    partial6 += p1 * v[6];
    partial7 += p1 * v[7];
  }

  const float out0 = WarpReduceSum(partial0);
  const float out1 = WarpReduceSum(partial1);
  const float out2 = WarpReduceSum(partial2);
  const float out3 = WarpReduceSum(partial3);
  const float out4 = WarpReduceSum(partial4);
  const float out5 = WarpReduceSum(partial5);
  const float out6 = WarpReduceSum(partial6);
  const float out7 = WarpReduceSum(partial7);

  if (lane == 0) {
    output[output_base + 0] = out0;
    output[output_base + 1] = out1;
    output[output_base + 2] = out2;
    output[output_base + 3] = out3;
    output[output_base + 4] = out4;
    output[output_base + 5] = out5;
    output[output_base + 6] = out6;
    output[output_base + 7] = out7;
  }
}

}  // namespace

extern "C" void LaunchMusaCausalAttentionForwardFloat(
    const float* query, const float* key, const float* value, const float* scale,
    float* softmax, float* output, int64_t groups, int query_dim, int key_dim,
    int head_dim, musaStream_t stream) {
  if (groups <= 0 || query_dim <= 0 || key_dim <= 0 || head_dim != 8 ||
      query_dim > key_dim || key_dim > 64) {
    return;
  }
  const dim3 block(kWarpSize, kRowsPerBlock, 1);
  const dim3 grid((query_dim + kRowsPerBlock - 1) / kRowsPerBlock, groups, 1);
  CausalAttentionForwardDh8FloatKernel<<<grid, block, 0, stream>>>(
      query, key, value, scale, softmax, output, groups, query_dim, key_dim);
}

}  // namespace musa
}  // namespace tensorflow
