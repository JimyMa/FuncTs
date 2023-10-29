#include "functs/csrc/jit/ir/symbol_ext.h"
#include <ATen/Context.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
at::Tensor Access(at::Tensor src) { return src.clone(); }

at::Tensor Assign(at::Tensor self, at::Tensor src, c10::optional<bool> n) {
  std::cout << "assing" << std::endl;
  return src.clone();
}

at::Tensor ImmutSelect(at::Tensor src, int64_t dim, int64_t index) {
  std::cout << "select" << std::endl;
  return src.select(dim, index).clone();
}

at::Tensor ImmutSelectRev(at::Tensor self, at::Tensor src, int64_t dim,
                          int64_t index) {
  std::cout << "select_rev" << std::endl;
  auto immut_self = self.clone();
  immut_self.select(dim, index).copy_(src);
  return immut_self;
}

at::Tensor ImmutSlice(at::Tensor src, int64_t dim,
                      c10::optional<int64_t> start = 0,
                      c10::optional<int64_t> end = 0, int64_t step = 1) {
  std::cout << "slice" << std::endl;
  return src.slice(dim, start, end, step).clone();
}
at::Tensor ImmutSliceRev(at::Tensor self, at::Tensor src, int64_t dim,
                         c10::optional<int64_t> start = 0,
                         c10::optional<int64_t> end = 0, int64_t step = 1) {
  std::cout << "slice_rev" << std::endl;
  auto immut_self = self.clone();
  immut_self.slice(dim, start, end, step).copy_(src);
  return immut_self;
}

at::Tensor ImmutSqueeze(at::Tensor src, int64_t dim) {
  std::cout << "immut_squeeze" << std::endl;
  return src.squeeze(dim).clone();
}
at::Tensor ImmutUnqueeze(at::Tensor src, int64_t dim) {
  std::cout << "immut_unsqueeze" << std::endl;
  return src.unsqueeze(dim).clone();
}

static auto _registry =
    RegisterOperators()
        .op("immut::access(Tensor src) -> Tensor", Access,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::assign(Tensor self, Tensor src, bool? n=None) -> Tensor",
            Assign,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::select(Tensor src, int dim, int index) -> Tensor",
            ImmutSelect,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::select_rev(Tensor self, Tensor src, int dim, int index) "
            "-> Tensor",
            ImmutSelectRev,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? "
            "end=None, SymInt step=1) -> Tensor",
            ImmutSlice,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::slice_rev(Tensor self, Tensor src, int dim=0, SymInt? "
            "start=None, SymInt? end=None, SymInt step=1) -> Tensor",
            ImmutSliceRev,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::squeeze(Tensor self, int dim) -> Tensor", ImmutSqueeze,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::unsqueeze(Tensor self, int dim) -> Tensor", ImmutUnqueeze,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION));
;
static auto x = 1;
} // namespace jit
} // namespace torch
