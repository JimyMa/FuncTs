#pragma once

#include "ir_utils.h"

namespace torch {
namespace jit {
namespace tensorexpr {

using RegId = uint32_t;

struct alignas(32) DimInst {
  IRNodeType op = kPrimitive;
  std::array<RegId, 4> src;
  CompareSelectOperation cmp;
};

std::ostream &operator<<(std::ostream &os, const DimInst &inst);

using VarRegMap = std::unordered_map<VarPtr, RegId>;

class DimExprEvaluator {
 public:
  DimExprEvaluator(const ExprPtr &expr);
  int64_t evaluate(const std::unordered_map<VarPtr, int64_t> &args) const;
  void dump() const;

 private:
  uint32_t numRegs = 0;
  std::vector<DimInst> insts;
  std::vector<int64_t> constPool;
  VarRegMap varToReg;
};

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch