#include "functs/csrc/jit/ir/symbol_ext.h"
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
at::Tensor Access(at::Tensor src) { return src; }

at::Tensor Assign(at::Tensor self, at::Tensor src, bool n) { return self; }

at::Tensor ImmutSelect(at::Tensor src, int64_t dim, int64_t index) {
  return src;
}
at::Tensor ImmutSelectRev(at::Tensor self, at::Tensor src, int64_t dim,
                          int64_t index) {
  return self;
}

at::Tensor ImmutSlice(at::Tensor src, int64_t dim,
                      c10::optional<int64_t> start = 0,
                      c10::optional<int64_t> end = 0, int64_t step = 1) {
  return src;
}
at::Tensor ImmutSliceRev(at::Tensor self, at::Tensor src, int64_t dim,
                         c10::optional<int64_t> start = 0,
                         c10::optional<int64_t> end = 0, int64_t step = 1) {
  return self;
}

void profEnd(const at::Tensor src) {}
static auto _registry =
    RegisterOperators()
        .op("immut::access(Tensor src) -> Tensor", Access,
            RegisterOperators::options().aliasAnalysis(
                c10::AliasAnalysisKind::PURE_FUNCTION))
        .op("immut::assign(Tensor self, Tensor src, bool n) -> Tensor", Assign,
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
                c10::AliasAnalysisKind::PURE_FUNCTION));
static auto x = 1;
} // namespace jit
} // namespace torch
