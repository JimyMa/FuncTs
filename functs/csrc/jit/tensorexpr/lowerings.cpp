#include <functs/csrc/jit/tensorexpr/lowerings.h>
#include <functs/csrc/jit/tensorexpr/nnc_ext.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>

namespace torch {
namespace jit {
namespace tensorexpr {

void init_nnc_ext() {
  auto immut_select_fn = [](const std::vector<ArgValue> &inputs,
                            const std::vector<ExprHandle> &outputShape,
                            const std::vector<ExprHandle> &outputStrides,
                            const c10::optional<ScalarType> &outputType,
                            at::Device device) {
    return computeImmutSlice(inputs, outputShape);
  };
  const char *external_func_name = "immut_select";
  auto &te_lowering_registry = torch::jit::tensorexpr::getNNCLoweringRegistry();
  te_lowering_registry.insert(
      parseSchema("immut::slice(Tensor src, int dim=0, SymInt? start=None, "
                  "SymInt? end=None, SymInt step=1) -> Tensor"),
      immut_select_fn);
  auto &te_nnc_func_registry = getNNCFunctionRegistry();
  // te_nnc_func_registry[external_func_name] = immut_select_fn;
}

// struct RegisterNNCLoweringsFunction;

// static RegisterNNCLoweringsFunction immut_slice(
//     {"immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? "
//      "end=None, SymInt step=1) -> Tensor"},
//     );

} // namespace tensorexpr
} // namespace jit
} // namespace torch
