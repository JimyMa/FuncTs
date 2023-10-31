#pragma once

#include "torch/csrc/jit/tensorexpr/fwd_decls.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"

namespace torch {
namespace jit {
namespace tensorexpr {

class ExprReconstructor : public IRMutator {
 public:
  ExprReconstructor(std::unordered_map<VarPtr, ExprPtr>& varExprMap)
      : varExprMap_(varExprMap) {}

  ExprPtr mutate(VarPtr var) override {
    if (varExprMap_.count(var)) {
      return varExprMap_[var];
    }
    return var;
  }

 private:
  std::unordered_map<VarPtr, ExprPtr>& varExprMap_;
};

class LetVarAnalysis : public IRVisitor {
 public:
  LetVarAnalysis(std::unordered_map<VarPtr, ExprPtr>& varExprMap)
      : varExprMap_(varExprMap) {}

  void visit(LetPtr let) override {
    ExprReconstructor reconstructor(varExprMap_);
    auto value = let->value()->accept_mutator(&reconstructor);
    varExprMap_.insert({let->var(), value});
  }

 private:
  std::unordered_map<VarPtr, ExprPtr>& varExprMap_;
};

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
