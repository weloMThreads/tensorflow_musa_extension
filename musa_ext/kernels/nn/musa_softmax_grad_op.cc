#include <musa_runtime.h>

#include <cstdint>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace musa {
namespace {

extern "C" void LaunchMusaSmallLastDimSoftmaxGradFloat(
    const float* softmax, const float* dy, float* dx, int64_t outer_size,
    int last_dim, musaStream_t stream);

class MusaSoftmaxGradOp : public MusaOpKernel {
 public:
  explicit MusaSoftmaxGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& softmax = ctx->input(0);
    const Tensor& dy = ctx->input(1);
    OP_REQUIRES(ctx, softmax.dtype() == DT_FLOAT && dy.dtype() == DT_FLOAT,
                errors::InvalidArgument("MusaSoftmaxGrad expects float"));
    OP_REQUIRES(ctx, softmax.shape() == dy.shape(),
                errors::InvalidArgument(
                    "MusaSoftmaxGrad input shape mismatch: ",
                    softmax.shape().DebugString(), " vs ",
                    dy.shape().DebugString()));
    OP_REQUIRES(ctx, softmax.dims() >= 1,
                errors::InvalidArgument("MusaSoftmaxGrad expects rank >= 1"));

    const int64_t last_dim_i64 = softmax.dim_size(softmax.dims() - 1);
    OP_REQUIRES(ctx, last_dim_i64 > 0 && last_dim_i64 <= 64,
                errors::InvalidArgument(
                    "MusaSoftmaxGrad supports last_dim in (0, 64], got ",
                    last_dim_i64));

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, softmax.shape(), &dx));
    if (dx->NumElements() == 0) return;

    const int64_t outer_size = dx->NumElements() / last_dim_i64;
    LaunchMusaSmallLastDimSoftmaxGradFloat(
        softmax.flat<float>().data(), dy.flat<float>().data(),
        dx->flat<float>().data(), outer_size, static_cast<int>(last_dim_i64),
        GetMusaStreamByCtx(ctx));
    const auto status = musaGetLastError();
    OP_REQUIRES(ctx, status == musaSuccess,
                errors::Internal("MusaSoftmaxGrad kernel failed: ",
                                 musaGetErrorString(status)));
  }
};

REGISTER_KERNEL_BUILDER(Name("MusaSoftmaxGrad").Device("MUSA"),
                        MusaSoftmaxGradOp);

}  // namespace
}  // namespace musa

REGISTER_OP("MusaSoftmaxGrad")
    .Input("softmax: float")
    .Input("dy: float")
    .Output("dx: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

}  // namespace tensorflow
