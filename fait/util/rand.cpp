#include "rand.h"

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ops/rand.h>
#include <ATen/ops/randint.h>

namespace torch {
namespace jit {

static auto rng = at::make_generator<at::CUDAGeneratorImpl>();

at::Tensor generateRandomTensor(TensorTypePtr type) {
  auto tensorTy = type->cast<TensorType>();
  auto shape = *tensorTy->sizes().concrete_sizes();
  auto dtype = *tensorTy->scalarType();
  switch (*tensorTy->scalarType()) {
    case c10::kFloat:
      return at::rand(shape, rng, c10::kFloat, c10::kStrided, c10::kCUDA,
                      c10::nullopt);

    case c10::kLong:
      return at::randint(0, 5, shape, rng, c10::kLong, c10::kStrided,
                         c10::kCUDA, c10::nullopt);

    case c10::kBool:
      return at::randint(0, 2, shape, rng, c10::kBool, c10::kStrided,
                         c10::kCUDA, c10::nullopt);

    default:
      TORCH_CHECK(false, "Dtype ", dtype, " not supported");
  }
}

}  // namespace jit
}  // namespace torch
