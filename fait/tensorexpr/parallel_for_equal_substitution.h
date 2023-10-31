#ifndef LONG_TAIL_PARALLEL_FOR_EQUAL_SUBSTITUTION_H
#define LONG_TAIL_PARALLEL_FOR_EQUAL_SUBSTITUTION_H

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"

using namespace torch::jit::tensorexpr;

class ParallelForEqualSubstitutionMutator : public IRMutator {
 public:
  ParallelForEqualSubstitutionMutator() {}
  ~ParallelForEqualSubstitutionMutator() override = default;

  StmtPtr mutate(ForPtr v) override;
  StmtPtr mutate(BlockPtr v) override;
  ExprPtr mutate(CompareSelectPtr v) override;

 private:
  bool mutate_state_ = false;
};

class TORCH_API ParallelForEqualSubstitution {
 public:
  static StmtPtr run(StmtPtr s);
};

#endif  // LONG_TAIL_PARALLEL_FOR_EQUAL_SUBSTITUTION_H