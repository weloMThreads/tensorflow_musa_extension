#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils/logging.h"

extern "C" {
void LaunchRmsNormFloat(const float* x, const float* gamma, float* y,
                        float* normalized, float* inv_rms, int64_t num_rows,
                        int64_t row_size, float epsilon, musaStream_t stream);
void LaunchRmsNormGradDxFloat(const float* x, const float* inv_rms,
                              const float* gamma, const float* dy,
                              float* dx, int64_t num_rows,
                              int64_t row_size, musaStream_t stream);
}

namespace tensorflow {
namespace musa {

class MusaRmsNormOp : public MusaOpKernel {
 public:
  explicit MusaRmsNormOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& gamma = ctx->input(1);

    OP_REQUIRES(ctx, x.dims() >= 1,
                errors::InvalidArgument("Input rank must be >= 1"));
    const int64_t row_size = x.dim_size(x.dims() - 1);
    OP_REQUIRES(ctx, row_size > 0,
                errors::InvalidArgument("Last dimension must be positive"));
    OP_REQUIRES(
        ctx, gamma.NumElements() == row_size,
        errors::InvalidArgument("Gamma size mismatch: expected ", row_size,
                                ", got ", gamma.NumElements()));

    Tensor* y = nullptr;
    Tensor* normalized = nullptr;
    Tensor* inv_rms = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, x.shape(), &normalized));

    TensorShape inv_shape;
    for (int i = 0; i < x.dims() - 1; ++i) {
      inv_shape.AddDim(x.dim_size(i));
    }
    inv_shape.AddDim(1);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, inv_shape, &inv_rms));

    if (x.NumElements() == 0) return;

    const int64_t num_rows = x.NumElements() / row_size;
    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    LaunchRmsNormFloat(x.flat<float>().data(), gamma.flat<float>().data(),
                       y->flat<float>().data(),
                       normalized->flat<float>().data(),
                       inv_rms->flat<float>().data(), num_rows, row_size,
                       epsilon_, stream);
  }

 private:
  float epsilon_;
};


class MusaRmsNormGradDxOp : public MusaOpKernel {
 public:
  explicit MusaRmsNormGradDxOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& inv_rms = ctx->input(1);
    const Tensor& gamma = ctx->input(2);
    const Tensor& dy = ctx->input(3);

    OP_REQUIRES(ctx, x.dtype() == DT_FLOAT && inv_rms.dtype() == DT_FLOAT &&
                         gamma.dtype() == DT_FLOAT && dy.dtype() == DT_FLOAT,
                errors::InvalidArgument("MusaRmsNormGradDx expects float inputs"));
    OP_REQUIRES(ctx, x.shape() == dy.shape(),
                errors::InvalidArgument("MusaRmsNormGradDx x/dy shape mismatch"));
    OP_REQUIRES(ctx, gamma.dims() == 1,
                errors::InvalidArgument("MusaRmsNormGradDx gamma must be rank 1"));

    const int64_t row_size = gamma.NumElements();
    OP_REQUIRES(ctx, row_size > 0 && x.NumElements() % row_size == 0,
                errors::InvalidArgument("MusaRmsNormGradDx invalid row size"));
    const int64_t num_rows = x.NumElements() / row_size;
    OP_REQUIRES(ctx, inv_rms.NumElements() == num_rows,
                errors::InvalidArgument("MusaRmsNormGradDx inv size mismatch"));

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    if (x.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    LaunchRmsNormGradDxFloat(
        x.flat<float>().data(), inv_rms.flat<float>().data(),
        gamma.flat<float>().data(), dy.flat<float>().data(),
        dx->flat<float>().data(), num_rows, row_size,
        reinterpret_cast<musaStream_t>(handle.GetStream()));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MusaRmsNorm").Device("MUSA").TypeConstraint<float>("T"),
    MusaRmsNormOp);

REGISTER_KERNEL_BUILDER(
    Name("MusaRmsNormGradDx").Device("MUSA").TypeConstraint<float>("T"),
    MusaRmsNormGradDxOp);

}  // namespace musa

REGISTER_OP("MusaRmsNormGradDx")
    .Input("x: T")
    .Input("inv_rms: T")
    .Input("gamma: T")
    .Input("dy: T")
    .Output("dx: T")
    .Attr("T: {float}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::OkStatus();
    });

REGISTER_OP("MusaRmsNorm")
    .Input("x: T")
    .Input("gamma: T")
    .Output("y: T")
    .Output("normalized: T")
    .Output("inv_rms: T")
    .Attr("T: {float}")
    .Attr("epsilon: float = 0.00001")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle x = c->input(0);
      c->set_output(0, x);
      c->set_output(1, x);
      if (!c->RankKnown(x)) {
        c->set_output(2, c->UnknownShape());
        return ::tensorflow::OkStatus();
      }
      std::vector<::tensorflow::shape_inference::DimensionHandle> dims;
      const int rank = c->Rank(x);
      dims.reserve(rank);
      for (int i = 0; i < rank - 1; ++i) {
        dims.push_back(c->Dim(x, i));
      }
      dims.push_back(c->MakeDim(1));
      c->set_output(2, c->MakeShape(dims));
      return ::tensorflow::OkStatus();
    });

}  // namespace tensorflow
