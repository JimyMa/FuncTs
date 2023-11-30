#include <functs/csrc/jit/ir/symbol_ext.h>
#include <functs/csrc/jit/tensorexpr/lowerings.h>
#include <functs/csrc/jit/tensorexpr/nnc_ext.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>

namespace torch {
namespace jit {
namespace tensorexpr {

void registerNNCImmutLoweringFunction(
    const char* schema,
    NNCLoweringFunction func) {
  auto& te_lowering_registry = torch::jit::tensorexpr::getNNCLoweringRegistry();
  te_lowering_registry.insert(parseSchema(schema), func);
  auto& custom_opset = getCustomOperatorSet();
  custom_opset.insert({schema});
}

void init_nnc_ext() {
  const char* immutAccessSchema = "immut::access(Tensor src) -> Tensor";
  const char* immutAssignSchema =
      "immut::assign(Tensor self, Tensor src, bool? n=None) -> Tensor";
  auto immut_access_assign_fn = [](const std::vector<ArgValue>& inputs,
                                   const std::vector<ExprHandle>& outputShape,
                                   const std::vector<ExprHandle>& outputStrides,
                                   const c10::optional<ScalarType>& outputType,
                                   at::Device device) {
    return computeImmutAssign(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutAssignSchema, immut_access_assign_fn);

  const char* immutCloneSchema =
      "aten::clone(Tensor self, *, MemoryFormat? "
      "memory_format=None) -> Tensor ";
  auto immut_clone_fn = [](const std::vector<ArgValue>& inputs,
                           const std::vector<ExprHandle>& outputShape,
                           const std::vector<ExprHandle>& outputStrides,
                           const c10::optional<ScalarType>& outputType,
                           at::Device device) {
    return computeClone(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutCloneSchema, immut_clone_fn);
  registerNNCImmutLoweringFunction(immutAccessSchema, immut_clone_fn);

  const char* immutSelectSchema =
      "immut::select(Tensor src, int dim, int index) -> Tensor";
  auto immut_select_fn = [](const std::vector<ArgValue>& inputs,
                            const std::vector<ExprHandle>& outputShape,
                            const std::vector<ExprHandle>& outputStrides,
                            const c10::optional<ScalarType>& outputType,
                            at::Device device) {
    return computeImmutSelect(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutSelectSchema, immut_select_fn);

  const char* immutSelectRevSchema =
      "immut::select_rev(Tensor self, Tensor "
      "src, int dim, int index) -> Tensor";
  auto immut_select_rev_fn = [](const std::vector<ArgValue>& inputs,
                                const std::vector<ExprHandle>& outputShape,
                                const std::vector<ExprHandle>& outputStrides,
                                const c10::optional<ScalarType>& outputType,
                                at::Device device) {
    return computeImmutSelectRev(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutSelectRevSchema, immut_select_rev_fn);

  const char* immutSliceSchema =
      "immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? "
      "end=None, SymInt step=1) -> Tensor";
  auto immut_slice_fn = [](const std::vector<ArgValue>& inputs,
                           const std::vector<ExprHandle>& outputShape,
                           const std::vector<ExprHandle>& outputStrides,
                           const c10::optional<ScalarType>& outputType,
                           at::Device device) {
    return computeImmutSlice(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutSliceSchema, immut_slice_fn);

  const char* immutSliceRevSchema =
      "immut::slice_rev(Tensor self, Tensor src, int "
      "dim=0, SymInt? start=None, SymInt? "
      "end=None, SymInt step=1) -> Tensor";
  auto immut_slice_rev_fn = [](const std::vector<ArgValue>& inputs,
                               const std::vector<ExprHandle>& outputShape,
                               const std::vector<ExprHandle>& outputStrides,
                               const c10::optional<ScalarType>& outputType,
                               at::Device device) {
    return computeImmutSliceRev(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutSliceRevSchema, immut_slice_rev_fn);

  const char* immutUnsqueezeSchema =
      "immut::unsqueeze(Tensor self, int dim) -> Tensor";
  auto immut_unsqueeze_fn = [](const std::vector<ArgValue>& inputs,
                               const std::vector<ExprHandle>& outputShape,
                               const std::vector<ExprHandle>& outputStrides,
                               const c10::optional<ScalarType>& outputType,
                               at::Device device) {
    return computeImmutUnsqueeze(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutUnsqueezeSchema, immut_unsqueeze_fn);

  const char* immutViewSchema =
      "immut::view(Tensor self, int[] size) -> Tensor";
  const char* immutReshapeSchema =
      "immut::reshape(Tensor self, int[] size) -> Tensor";
  auto immut_view_fn = [](const std::vector<ArgValue>& inputs,
                          const std::vector<ExprHandle>& outputShape,
                          const std::vector<ExprHandle>& outputStrides,
                          const c10::optional<ScalarType>& outputType,
                          at::Device device) {
    return computeImmutView(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutViewSchema, immut_view_fn);
  registerNNCImmutLoweringFunction(immutReshapeSchema, immut_view_fn);

  const char* immutRepeatSchema =
      "immut::repeat(Tensor self, int[] size) -> Tensor";
  const char* immutExpandSchema =
      "immut::expand(Tensor self, int[] size, *, bool implicit) -> Tensor";
  const char* immutExpandAsSchema =
      "immut::expand_as(Tensor self, Tensor other) -> Tensor";
  auto immut_repeat_fn = [](const std::vector<ArgValue>& inputs,
                            const std::vector<ExprHandle>& outputShape,
                            const std::vector<ExprHandle>& outputStrides,
                            const c10::optional<ScalarType>& outputType,
                            at::Device device) {
    return computeImmutRepeat(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutRepeatSchema, immut_repeat_fn);
  registerNNCImmutLoweringFunction(immutExpandSchema, immut_repeat_fn);
  registerNNCImmutLoweringFunction(immutExpandAsSchema, immut_repeat_fn);
  const char* immutPermuteSchema =
      "immut::permute(Tensor self, int[] sizes) -> Tensor";
  auto immut_permute_fn = [](const std::vector<ArgValue>& inputs,
                             const std::vector<ExprHandle>& outputShape,
                             const std::vector<ExprHandle>& outputStrides,
                             const c10::optional<ScalarType>& outputType,
                             at::Device device) {
    return computeImmutPermute(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutPermuteSchema, immut_permute_fn);

  const char* immutIndexSchema =
      "immut::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor";
  auto immut_index_fn = [](const std::vector<ArgValue>& inputs,
                           const std::vector<ExprHandle>& outputShape,
                           const std::vector<ExprHandle>& outputStrides,
                           const c10::optional<ScalarType>& outputType,
                           at::Device device) {
    return computeImmutIndex(inputs, outputShape);
  };
  registerNNCImmutLoweringFunction(immutIndexSchema, immut_index_fn);

  // const char* tensorSchema =
  //     "aten::tensor.int(int t, *, ScalarType? dtype=None, Device?
  //     device=None, bool requires_grad=False) -> Tensor";
  // auto tensor_fn = [](const std::vector<ArgValue>& inputs,
  //                     const std::vector<ExprHandle>& outputShape,
  //                     const std::vector<ExprHandle>& outputStrides,
  //                     const c10::optional<ScalarType>& outputType,
  //                     at::Device device) {
  //   return computeTensor(inputs, outputShape);
  // };
  // registerNNCImmutLoweringFunction(tensorSchema, tensor_fn);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
