#ifndef LONG_TAIL_EVALUATE_OUTPUT_SHAPE_H
#define LONG_TAIL_EVALUATE_OUTPUT_SHAPE_H

#include "torch/csrc/jit/tensorexpr/fwd_decls.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"

using namespace torch::jit::tensorexpr;

class TORCH_API OutputShapeEvaluator : public IRVisitor {
 public:
  OutputShapeEvaluator(const std::unordered_map<VarPtr, int64_t>& dims_map,
                       int64_t degree)
      : dims_map_(dims_map), degree_(degree) {}
  ~OutputShapeEvaluator() override = default;

  void visit(VarPtr v) override;
  void visit(LongImmPtr v) override;
  void visit(BoolImmPtr v) override;
  void visit(IntImmPtr v) override;
  void visit(AddPtr v) override;
  void visit(SubPtr v) override;
  void visit(MulPtr v) override;
  void visit(DivPtr v) override;
  void visit(ModPtr v) override;
  void visit(MaxPtr v) override;
  void visit(MinPtr v) override;
  void visit(CompareSelectPtr v) override;
  void visit(IfThenElsePtr v) override;

  int64_t get_value() { return _tmp_value; }

 private:
  int64_t _tmp_value{-1};
  std::unordered_map<VarPtr, int64_t> dims_map_;
  int64_t degree_;
};

class TORCH_API EvaluateOutputShape {
 public:
  static int64_t run(ExprPtr s, /* Functor Statement */
                     const std::unordered_map<VarPtr, int64_t>& dims_map,
                     int64_t degree) {
    OutputShapeEvaluator evaluator(dims_map, degree);
    s->accept(&evaluator);
    return evaluator.get_value();
  }
};

#endif  // LONG_TAIL_EVALUATE_OUTPUT_SHAPE_H
