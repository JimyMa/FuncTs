#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {

static RegisterOperators const reg({
    Operator(
        "immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? "
        "end=None, SymInt step=1) -> Tensor",
        [](Stack &stack) {
          std::cout << "??zdfffffffffffffffffffffffff" << std::endl;
          auto step = pop(stack);
          auto end = pop(stack);
          auto start = pop(stack);
          auto dim = pop(stack);
          auto src = pop(stack).toTensor();
          push(stack, src.clone());
        },
        c10::AliasAnalysisKind::PURE_FUNCTION),
    Operator(
        "immut::select(Tensor self, int dim, int index) -> Tensor",
        [](Stack &stack) {
          std::cout << "sdfsdfsdfqqqqqqqqqqqqqqqqqqqq??" << std::endl;
          auto idx = pop(stack);
          auto dim = pop(stack);
          auto src = pop(stack).toTensor();
          push(stack, src.clone());
        },
        c10::AliasAnalysisKind::PURE_FUNCTION),
    Operator(
        "immut::assign(Tensor self, Tensor src, bool? n=None) -> Tensor",
        [](Stack &stack) {
          std::cout << "?sdfsdfsdfsdsdfqerrrrrrrrrrrrrrrrrrr?" << std::endl;
          auto idx = pop(stack);
          auto dim = pop(stack);
          auto src = pop(stack).toTensor();
          push(stack, src.clone());
        },
        c10::AliasAnalysisKind::PURE_FUNCTION),
});

}
} // namespace torch