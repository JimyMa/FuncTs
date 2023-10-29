#include <functs/csrc/jit/ir/symbol_ext.h>
#include <functs/csrc/jit/tensorexpr/lowerings.h>
#include <functs/csrc/jit/tensorexpr/nnc_ext.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>

namespace torch {
namespace jit {
namespace tensorexpr {

void registerNNCImmutLoweringFunction(const char *schema,
                                      NNCLoweringFunction func) {
  auto &te_lowering_registry = torch::jit::tensorexpr::getNNCLoweringRegistry();
  te_lowering_registry.insert(parseSchema(schema), func);
  auto &custom_opset = getCustomOperatorSet();
  custom_opset.insert({schema});
}

void init_nnc_ext() {
  const char *immutAccessSchema = "immut::access(Tensor src) -> Tensor";
  const char *immutAssignSchema =
      "immut::assign(Tensor self, Tensor src, bool? n=None) -> Tensor";
  auto immut_access_assign_fn = [](const std::vector<ArgValue> &inputs,
                                   const std::vector<ExprHandle> &outputShape,
                                   const std::vector<ExprHandle> &outputStrides,
                                   const c10::optional<ScalarType> &outputType,
                                   at::Device device) {
    return computeImmutAssign(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutAccessSchema, immut_access_assign_fn);
  registerNNCImmutLoweringFunction(immutAssignSchema, immut_access_assign_fn);

  const char *immutSelectSchema =
      "immut::select(Tensor src, int dim, int index) -> Tensor";
  auto immut_select_fn = [](const std::vector<ArgValue> &inputs,
                            const std::vector<ExprHandle> &outputShape,
                            const std::vector<ExprHandle> &outputStrides,
                            const c10::optional<ScalarType> &outputType,
                            at::Device device) {
    return computeImmutSelect(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutSelectSchema, immut_select_fn);

  const char *immutSliceSchema =
      "immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? "
      "end=None, SymInt step=1) -> Tensor";
  auto immut_slice_fn = [](const std::vector<ArgValue> &inputs,
                           const std::vector<ExprHandle> &outputShape,
                           const std::vector<ExprHandle> &outputStrides,
                           const c10::optional<ScalarType> &outputType,
                           at::Device device) {
    return computeImmutSlice(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutSliceSchema, immut_slice_fn);
}

// struct RegisterNNCLoweringsFunction;

// static RegisterNNCLoweringsFunction immut_slice(
//     {"immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? "
//      "end=None, SymInt step=1) -> Tensor"},
//     );

} // namespace tensorexpr
} // namespace jit
} // namespace torch
