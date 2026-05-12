#include <math.h>
#include <stdint.h>

#include <musa_runtime.h>

namespace tensorflow {
namespace musa {
namespace {

constexpr int kMaxDim = 64;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
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

__global__ void CausalAttentionGradFloatKernel(
    const float* __restrict__ softmax, const float* __restrict__ dout,
    const float* __restrict__ query, const float* __restrict__ key,
    const float* __restrict__ value, const float* __restrict__ scale_ptr,
    float* __restrict__ dquery, float* __restrict__ dkey,
    float* __restrict__ dvalue, int64_t groups, int query_dim, int key_dim,
    int value_dim) {
  const int64_t group = static_cast<int64_t>(blockIdx.x);
  if (group >= groups) return;

  __shared__ float work[kMaxDim * kMaxDim];
  __shared__ float row_dot[kMaxDim];

  for (int q = threadIdx.x; q < query_dim; q += blockDim.x) {
    row_dot[q] = 0.0f;
  }
  __syncthreads();

  const float scale = *scale_ptr;
  const int64_t softmax_base = group * static_cast<int64_t>(query_dim) * key_dim;
  const int64_t q_base = group * static_cast<int64_t>(query_dim) * value_dim;
  const int64_t k_base = group * static_cast<int64_t>(key_dim) * value_dim;
  const int64_t v_base = k_base;

  for (int idx = threadIdx.x; idx < query_dim * key_dim; idx += blockDim.x) {
    const int q = idx / key_dim;
    const int k = idx - q * key_dim;
    const int last_unmasked_key = key_dim - query_dim + q;
    float ds = 0.0f;
    if (k <= last_unmasked_key) {
      for (int d = 0; d < value_dim; ++d) {
        ds += dout[q_base + static_cast<int64_t>(q) * value_dim + d] *
              value[v_base + static_cast<int64_t>(k) * value_dim + d];
      }
      const float s = softmax[softmax_base + idx];
      atomicAdd(&row_dot[q], ds * s);
    }
    work[idx] = ds;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < query_dim * key_dim; idx += blockDim.x) {
    const int q = idx / key_dim;
    const int k = idx - q * key_dim;
    const int last_unmasked_key = key_dim - query_dim + q;
    float dlogits = 0.0f;
    if (k <= last_unmasked_key) {
      const float s = softmax[softmax_base + idx];
      dlogits = (work[idx] - row_dot[q]) * s * scale;
    }
    work[idx] = dlogits;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < query_dim * value_dim; idx += blockDim.x) {
    const int q = idx / value_dim;
    const int d = idx - q * value_dim;
    float sum = 0.0f;
    const int last_unmasked_key = key_dim - query_dim + q;
    for (int k = 0; k <= last_unmasked_key; ++k) {
      sum += work[q * key_dim + k] *
             key[k_base + static_cast<int64_t>(k) * value_dim + d];
    }
    dquery[q_base + static_cast<int64_t>(q) * value_dim + d] = sum;
  }

  for (int idx = threadIdx.x; idx < key_dim * value_dim; idx += blockDim.x) {
    const int k = idx / value_dim;
    const int d = idx - k * value_dim;
    float dk_sum = 0.0f;
    float dv_sum = 0.0f;
    for (int q = 0; q < query_dim; ++q) {
      const int last_unmasked_key = key_dim - query_dim + q;
      if (k <= last_unmasked_key) {
        dk_sum += work[q * key_dim + k] *
                  query[q_base + static_cast<int64_t>(q) * value_dim + d];
        dv_sum += softmax[softmax_base + static_cast<int64_t>(q) * key_dim + k] *
                  dout[q_base + static_cast<int64_t>(q) * value_dim + d];
      }
    }
    dkey[k_base + static_cast<int64_t>(k) * value_dim + d] = dk_sum;
    dvalue[v_base + static_cast<int64_t>(k) * value_dim + d] = dv_sum;
  }
}


__global__ void CausalAttentionGradDh8RecomputeFloatKernel(
    const float* __restrict__ softmax, const float* __restrict__ dout,
    const float* __restrict__ query, const float* __restrict__ key,
    const float* __restrict__ value, const float* __restrict__ scale_ptr,
    float* __restrict__ dquery, float* __restrict__ dkey,
    float* __restrict__ dvalue, int64_t groups, int query_dim, int key_dim) {
  (void)softmax;
  const int64_t group = static_cast<int64_t>(blockIdx.x);
  if (group >= groups) return;

  constexpr int kValueDim = 8;
  constexpr int kMaxRecomputeDim = 40;
  constexpr int kMaxRecomputeElems = kMaxRecomputeDim * kMaxRecomputeDim;
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

  __shared__ float dlogits[kMaxRecomputeElems];
  __shared__ float probs[kMaxRecomputeElems];

  const float scale = *scale_ptr;
  const int64_t q_base = group * static_cast<int64_t>(query_dim) * kValueDim;
  const int64_t k_base = group * static_cast<int64_t>(key_dim) * kValueDim;
  const int64_t v_base = k_base;

  for (int q = warp_id; q < query_dim; q += kWarpsPerBlock) {
    const int last_unmasked_key = key_dim - query_dim + q;
    const int key_idx0 = lane;
    const int key_idx1 = lane + kWarpSize;
    const int64_t q_off = q_base + static_cast<int64_t>(q) * kValueDim;

    const float q0 = query[q_off + 0];
    const float q1 = query[q_off + 1];
    const float q2 = query[q_off + 2];
    const float q3 = query[q_off + 3];
    const float q4 = query[q_off + 4];
    const float q5 = query[q_off + 5];
    const float q6 = query[q_off + 6];
    const float q7 = query[q_off + 7];

    auto dot_key = [&](int key_idx) {
      const float* key_ptr =
          key + k_base + static_cast<int64_t>(key_idx) * kValueDim;
      return (q0 * key_ptr[0] + q1 * key_ptr[1] + q2 * key_ptr[2] +
              q3 * key_ptr[3] + q4 * key_ptr[4] + q5 * key_ptr[5] +
              q6 * key_ptr[6] + q7 * key_ptr[7]) *
             scale;
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

    const float* dout_ptr = dout + q_off;
    auto dot_value = [&](int key_idx) {
      const float* value_ptr =
          value + v_base + static_cast<int64_t>(key_idx) * kValueDim;
      return dout_ptr[0] * value_ptr[0] + dout_ptr[1] * value_ptr[1] +
             dout_ptr[2] * value_ptr[2] + dout_ptr[3] * value_ptr[3] +
             dout_ptr[4] * value_ptr[4] + dout_ptr[5] * value_ptr[5] +
             dout_ptr[6] * value_ptr[6] + dout_ptr[7] * value_ptr[7];
    };

    const float ds0 = valid0 ? dot_value(key_idx0) : 0.0f;
    const float ds1 = valid1 ? dot_value(key_idx1) : 0.0f;
    const float row_dot = WarpReduceSum(ds0 * p0 + ds1 * p1);
    const int row_base = q * key_dim;
    if (key_idx0 < key_dim) {
      probs[row_base + key_idx0] = p0;
      dlogits[row_base + key_idx0] =
          valid0 ? (ds0 - row_dot) * p0 * scale : 0.0f;
    }
    if (key_idx1 < key_dim) {
      probs[row_base + key_idx1] = p1;
      dlogits[row_base + key_idx1] =
          valid1 ? (ds1 - row_dot) * p1 * scale : 0.0f;
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < query_dim * kValueDim; idx += blockDim.x) {
    const int q = idx / kValueDim;
    const int d = idx - q * kValueDim;
    float sum = 0.0f;
    const int last_unmasked_key = key_dim - query_dim + q;
    for (int k = 0; k <= last_unmasked_key; ++k) {
      sum += dlogits[q * key_dim + k] *
             key[k_base + static_cast<int64_t>(k) * kValueDim + d];
    }
    dquery[q_base + static_cast<int64_t>(q) * kValueDim + d] = sum;
  }

  for (int idx = threadIdx.x; idx < key_dim * kValueDim; idx += blockDim.x) {
    const int k = idx / kValueDim;
    const int d = idx - k * kValueDim;
    float dk_sum = 0.0f;
    float dv_sum = 0.0f;
    for (int q = 0; q < query_dim; ++q) {
      const int last_unmasked_key = key_dim - query_dim + q;
      if (k <= last_unmasked_key) {
        dk_sum += dlogits[q * key_dim + k] *
                  query[q_base + static_cast<int64_t>(q) * kValueDim + d];
        dv_sum += probs[q * key_dim + k] *
                  dout[q_base + static_cast<int64_t>(q) * kValueDim + d];
      }
    }
    dkey[k_base + static_cast<int64_t>(k) * kValueDim + d] = dk_sum;
    dvalue[v_base + static_cast<int64_t>(k) * kValueDim + d] = dv_sum;
  }
}

template <int kValueDim>
__global__ void CausalAttentionGradFixedValueDimFloatKernel(
    const float* __restrict__ softmax, const float* __restrict__ dout,
    const float* __restrict__ query, const float* __restrict__ key,
    const float* __restrict__ value, const float* __restrict__ scale_ptr,
    float* __restrict__ dquery, float* __restrict__ dkey,
    float* __restrict__ dvalue, int64_t groups, int query_dim, int key_dim) {
  const int64_t group = static_cast<int64_t>(blockIdx.x);
  if (group >= groups) return;

  __shared__ float work[kMaxDim * kMaxDim];
  __shared__ float row_dot[kMaxDim];

  for (int q = threadIdx.x; q < query_dim; q += blockDim.x) {
    row_dot[q] = 0.0f;
  }
  __syncthreads();

  const float scale = *scale_ptr;
  const int64_t softmax_base = group * static_cast<int64_t>(query_dim) * key_dim;
  const int64_t q_base = group * static_cast<int64_t>(query_dim) * kValueDim;
  const int64_t k_base = group * static_cast<int64_t>(key_dim) * kValueDim;
  const int64_t v_base = k_base;

  for (int idx = threadIdx.x; idx < query_dim * key_dim; idx += blockDim.x) {
    const int q = idx / key_dim;
    const int k = idx - q * key_dim;
    const int last_unmasked_key = key_dim - query_dim + q;
    float ds = 0.0f;
    if (k <= last_unmasked_key) {
#pragma unroll
      for (int d = 0; d < kValueDim; ++d) {
        ds += dout[q_base + static_cast<int64_t>(q) * kValueDim + d] *
              value[v_base + static_cast<int64_t>(k) * kValueDim + d];
      }
      const float s = softmax[softmax_base + idx];
      atomicAdd(&row_dot[q], ds * s);
    }
    work[idx] = ds;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < query_dim * key_dim; idx += blockDim.x) {
    const int q = idx / key_dim;
    const int k = idx - q * key_dim;
    const int last_unmasked_key = key_dim - query_dim + q;
    float dlogits = 0.0f;
    if (k <= last_unmasked_key) {
      const float s = softmax[softmax_base + idx];
      dlogits = (work[idx] - row_dot[q]) * s * scale;
    }
    work[idx] = dlogits;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < query_dim * kValueDim; idx += blockDim.x) {
    const int q = idx / kValueDim;
    const int d = idx - q * kValueDim;
    float sum = 0.0f;
    const int last_unmasked_key = key_dim - query_dim + q;
    for (int k = 0; k <= last_unmasked_key; ++k) {
      sum += work[q * key_dim + k] *
             key[k_base + static_cast<int64_t>(k) * kValueDim + d];
    }
    dquery[q_base + static_cast<int64_t>(q) * kValueDim + d] = sum;
  }

  for (int idx = threadIdx.x; idx < key_dim * kValueDim; idx += blockDim.x) {
    const int k = idx / kValueDim;
    const int d = idx - k * kValueDim;
    float dk_sum = 0.0f;
    float dv_sum = 0.0f;
    for (int q = 0; q < query_dim; ++q) {
      const int last_unmasked_key = key_dim - query_dim + q;
      if (k <= last_unmasked_key) {
        dk_sum += work[q * key_dim + k] *
                  query[q_base + static_cast<int64_t>(q) * kValueDim + d];
        dv_sum += softmax[softmax_base + static_cast<int64_t>(q) * key_dim + k] *
                  dout[q_base + static_cast<int64_t>(q) * kValueDim + d];
      }
    }
    dkey[k_base + static_cast<int64_t>(k) * kValueDim + d] = dk_sum;
    dvalue[v_base + static_cast<int64_t>(k) * kValueDim + d] = dv_sum;
  }
}

}  // namespace

extern "C" void LaunchMusaCausalAttentionGradFloat(
    const float* softmax, const float* dout, const float* query,
    const float* key, const float* value, const float* scale, float* dquery,
    float* dkey, float* dvalue, int64_t groups, int query_dim, int key_dim,
    int value_dim, musaStream_t stream) {
  if (groups <= 0 || query_dim <= 0 || key_dim <= 0 || value_dim <= 0 ||
      query_dim > key_dim || query_dim > kMaxDim || key_dim > kMaxDim ||
      value_dim > kMaxDim) {
    return;
  }
  if (value_dim == 8 && query_dim <= 40 && key_dim <= 40) {
    CausalAttentionGradDh8RecomputeFloatKernel<<<groups, kThreads, 0, stream>>>(
        softmax, dout, query, key, value, scale, dquery, dkey, dvalue, groups,
        query_dim, key_dim);
    return;
  }
  if (value_dim == 8) {
    CausalAttentionGradFixedValueDimFloatKernel<8><<<groups, kThreads, 0, stream>>>(
        softmax, dout, query, key, value, scale, dquery, dkey, dvalue, groups,
        query_dim, key_dim);
    return;
  }
  CausalAttentionGradFloatKernel<<<groups, kThreads, 0, stream>>>(
      softmax, dout, query, key, value, scale, dquery, dkey, dvalue, groups,
      query_dim, key_dim, value_dim);
}

}  // namespace musa
}  // namespace tensorflow
