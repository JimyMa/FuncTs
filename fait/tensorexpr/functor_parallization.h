//
// Created by jimyma on 1/30/23.
//

#ifndef LONG_TAIL_FUNCTOR_PARALLIZATION_H
#define LONG_TAIL_FUNCTOR_PARALLIZATION_H

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"

using namespace torch::jit::tensorexpr;

class FunctorParallizationMutator : public IRMutator {
 public:
  FunctorParallizationMutator(
      int64_t degree, int64_t idx,
      std::unordered_map<BufPtr, std::vector<BufHandle>> buf_args_map,
      std::unordered_map<VarPtr, std::vector<VarHandle>> var_args_map,
      std::unordered_map<BufPtr, std::vector<BufHandle>> buf_ret_map)
      : degree_(degree),
        idx_(idx),
        buf_args_map_(buf_args_map),
        var_args_map_(var_args_map),
        buf_ret_map_(buf_ret_map) {}

  ExprPtr mutate(VarPtr v) override;
  ExprPtr mutate(BufPtr v) override;

 private:
  int64_t degree_;
  int64_t idx_;
  std::unordered_map<BufPtr, std::vector<BufHandle>> buf_args_map_;
  std::unordered_map<VarPtr, std::vector<VarHandle>> var_args_map_;
  std::unordered_map<BufPtr, std::vector<BufHandle>> buf_ret_map_;
};

class TORCH_API FunctorParallization {
 public:
  static StmtPtr Parallel_functor(
      StmtPtr s, int64_t degree, VarPtr iter_var,
      std::unordered_map<BufPtr, std::vector<BufHandle>> buf_args_map,
      std::unordered_map<VarPtr, std::vector<VarHandle>> var_args_map,
      std::unordered_map<BufPtr, std::vector<BufHandle>> buf_ret_map,
      std::unordered_map<VarPtr, std::vector<VarHandle>> dims_map);
};

class FunctorParallizationShapeMutator : public IRMutator {
 public:
  FunctorParallizationShapeMutator(
      int64_t degree, int64_t idx,
      std::unordered_map<VarPtr, std::vector<VarHandle>> dims_map)
      : degree_(degree), idx_(idx), dims_map_(dims_map) {}

  ExprPtr mutate(VarPtr v) override;
  ExprPtr mutate(BufPtr v) override;

 private:
  int64_t degree_;
  int64_t idx_;
  std::unordered_map<VarPtr, std::vector<VarHandle>> dims_map_;
};

#endif  // LONG_TAIL_FUNCTOR_PARALLIZATION_H
