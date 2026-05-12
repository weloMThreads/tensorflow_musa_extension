#include <musa_runtime.h>

#include "../utils_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace musa {
namespace {

extern "C" void LaunchMusaCausalAttentionGradFloat(
    const float* softmax, const float* dout, const float* query,
    const float* key, const float* value, const float* scale, float* dquery,
    float* dkey, float* dvalue, int64_t groups, int query_dim, int key_dim,
    int value_dim, musaStream_t stream);

class MusaCausalAttentionGradOp : public MusaOpKernel {
 public:
  explicit MusaCausalAttentionGradOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& softmax = ctx->input(0);
    const Tensor& dout = ctx->input(1);
    const Tensor& query = ctx->input(2);
    const Tensor& key = ctx->input(3);
    const Tensor& value = ctx->input(4);
    const Tensor& scale = ctx->input(5);
    OP_REQUIRES(ctx, softmax.dtype() == DT_FLOAT && dout.dtype() == DT_FLOAT &&
                         query.dtype() == DT_FLOAT && key.dtype() == DT_FLOAT &&
                         value.dtype() == DT_FLOAT && scale.dtype() == DT_FLOAT,
                errors::InvalidArgument("MusaCausalAttentionGrad expects float"));
    OP_REQUIRES(ctx, scale.NumElements() == 1,
                errors::InvalidArgument("MusaCausalAttentionGrad expects scalar scale"));
    OP_REQUIRES(ctx, softmax.dims() == 4 && dout.dims() == 4 &&
                         query.dims() == 4 && key.dims() == 4 &&
                         value.dims() == 4,
                errors::InvalidArgument("MusaCausalAttentionGrad expects rank-4 tensors"));

    const int64_t batch = softmax.dim_size(0);
    const int64_t heads = softmax.dim_size(1);
    const int64_t query_dim = softmax.dim_size(2);
    const int64_t key_dim = softmax.dim_size(3);
    const int64_t value_dim = value.dim_size(3);
    OP_REQUIRES(ctx, dout.dim_size(0) == batch && dout.dim_size(1) == heads &&
                         dout.dim_size(2) == query_dim &&
                         dout.dim_size(3) == value_dim &&
                         query.dim_size(0) == batch && query.dim_size(1) == heads &&
                         query.dim_size(2) == query_dim &&
                         query.dim_size(3) == value_dim &&
                         key.dim_size(0) == batch && key.dim_size(1) == heads &&
                         key.dim_size(2) == key_dim &&
                         key.dim_size(3) == value_dim &&
                         value.dim_size(0) == batch && value.dim_size(1) == heads &&
                         value.dim_size(2) == key_dim,
                errors::InvalidArgument("MusaCausalAttentionGrad incompatible shapes"));
    OP_REQUIRES(ctx, query_dim > 0 && key_dim > 0 && value_dim > 0 &&
                         query_dim <= key_dim && query_dim <= 64 &&
                         key_dim <= 64 && value_dim <= 64,
                errors::InvalidArgument("MusaCausalAttentionGrad invalid shape"));

    Tensor* dquery = nullptr;
    Tensor* dkey = nullptr;
    Tensor* dvalue = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, query.shape(), &dquery));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, key.shape(), &dkey));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, value.shape(), &dvalue));
    if (dquery->NumElements() == 0) return;

    LaunchMusaCausalAttentionGradFloat(
        softmax.flat<float>().data(), dout.flat<float>().data(),
        query.flat<float>().data(), key.flat<float>().data(),
        value.flat<float>().data(), scale.flat<float>().data(),
        dquery->flat<float>().data(), dkey->flat<float>().data(),
        dvalue->flat<float>().data(), batch * heads,
        static_cast<int>(query_dim), static_cast<int>(key_dim),
        static_cast<int>(value_dim), GetMusaStreamByCtx(ctx));
    const auto status = musaGetLastError();
    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal("MusaCausalAttentionGrad kernel failed: ",
                                 musaGetErrorString(status)));
  }
};

REGISTER_KERNEL_BUILDER(Name("MusaCausalAttentionGrad").Device("MUSA"),
                        MusaCausalAttentionGradOp);

}  // namespace
}  // namespace musa

REGISTER_OP("MusaCausalAttentionGrad")
    .Input("softmax: float")
    .Input("dout: float")
    .Input("query: float")
    .Input("key: float")
    .Input("value: float")
    .Input("scale: float")
    .Output("dquery: float")
    .Output("dkey: float")
    .Output("dvalue: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      c->set_output(1, c->input(3));
      c->set_output(2, c->input(4));
      return Status::OK();
    });

}  // namespace tensorflow
