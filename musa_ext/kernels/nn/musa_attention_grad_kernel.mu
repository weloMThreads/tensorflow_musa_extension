#include <stdint.h>

#include <musa_runtime.h>

namespace tensorflow {
namespace musa {
namespace {

constexpr int kMaxDim = 64;
constexpr int kThreads = 256;

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
