#include "evaluate_dim.h"

#include <iomanip>
#include <iostream>

#include "ir_utils.h"

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {

using ExprRegMap = std::unordered_map<ExprPtr, RegId, ExprHasher, ExprEq>;

class ConstCollector : public IRVisitor {
 public:
  ConstCollector(std::vector<int64_t>& pool, ExprRegMap& exprToReg)
      : pool(pool), exprToReg(exprToReg) {}

#define COLLECT_IMM(type)                 \
  void visit(type##ImmPtr imm) override { \
    auto expr = static_to<Expr>(imm);     \
    if (exprToReg.count(expr))            \
      return;                             \
    auto val = int64_t(imm->value());     \
    auto idx = pool.size();               \
    pool.push_back(val);                  \
    exprToReg.insert({expr, idx});        \
  }

  COLLECT_IMM(Int)
  COLLECT_IMM(Long)

#undef COLLECT_IMM

 private:
  std::vector<int64_t>& pool;
  ExprRegMap& exprToReg;
};

class VarCollector : public IRVisitor {
 public:
  VarCollector(VarRegMap& varToReg, ExprRegMap& exprToReg)
      : varToReg(varToReg), exprToReg(exprToReg) {}

  void visit(VarPtr var) override {
    if (varToReg.count(var))
      return;
    auto idx = exprToReg.size();
    varToReg.insert({var, idx});
    exprToReg.insert({static_to<Expr>(var), idx});
  }

 private:
  VarRegMap& varToReg;
  ExprRegMap& exprToReg;
};

class DimExprCompiler : public IRVisitor {
 public:
  DimExprCompiler(ExprRegMap& exprToReg, std::vector<DimInst>& insts)
      : exprToReg(exprToReg), insts(insts) {}

  void visit(LongImmPtr imm) override {}

  void visit(VarPtr var) override {}

#define COMPILE_BINARY_EXPR(op)                                      \
  void visit(op##Ptr node) override {                                \
    auto lhs = getRegFor(node->lhs()), rhs = getRegFor(node->rhs()); \
    nextReg(node);                                                   \
    insts.push_back({k##op, {lhs, rhs}});                            \
  }

  COMPILE_BINARY_EXPR(Add)
  COMPILE_BINARY_EXPR(Sub)
  COMPILE_BINARY_EXPR(Mul)
  COMPILE_BINARY_EXPR(Div)
  COMPILE_BINARY_EXPR(Mod)
  COMPILE_BINARY_EXPR(Max)
  COMPILE_BINARY_EXPR(Min)

#undef COMPILE_BINARY_EXPR

  void visit(CompareSelectPtr cmpSel) override {
    auto lhs = getRegFor(cmpSel->lhs()), rhs = getRegFor(cmpSel->rhs()),
         retVal1 = getRegFor(cmpSel->ret_val1()),
         retVal2 = getRegFor(cmpSel->ret_val2());
    nextReg(cmpSel);
    insts.push_back(
        {kCompareSelect,
         {lhs, rhs, retVal1, retVal2},
         cmpSel->compare_select_op()});
  }

  void visit(IfThenElsePtr ite) override {
    auto cond = getRegFor(ite->condition()),
         trueVal = getRegFor(ite->true_value()),
         falseVal = getRegFor(ite->false_value());
    nextReg(ite);
    insts.push_back({kOther, {cond, trueVal, falseVal}});
  }

  RegId getRegFor(ExprPtr expr) {
    auto it = exprToReg.find(expr);
    if (it != exprToReg.end())
      return it->second;
    expr->accept(this);
    it = exprToReg.find(expr);
    TORCH_CHECK(
        it != exprToReg.end(),
        "Cannot compile expression: ",
        std::to_string(expr));
    return it->second;
  }

 private:
  template <class NodePtr>
  RegId nextReg(const NodePtr& node) const {
    auto reg = exprToReg.size();
    exprToReg.insert({static_to<Expr>(node), reg});
    return reg;
  }

  ExprRegMap& exprToReg;
  std::vector<DimInst>& insts;
};

} // namespace

DimExprEvaluator::DimExprEvaluator(const ExprPtr& expr) {
  // Collect all constants and variables
  ExprRegMap exprToReg;
  ConstCollector constCol(constPool, exprToReg);
  expr->accept(&constCol);
  VarCollector varCol(varToReg, exprToReg);
  expr->accept(&varCol);

  // Compile expression
  DimExprCompiler compiler(exprToReg, insts);
  compiler.getRegFor(expr);
  numRegs = constPool.size() + varToReg.size() + insts.size();
}

int64_t DimExprEvaluator::evaluate(
    const std::unordered_map<VarPtr, int64_t>& args) const {
  // Create virtual registers
  std::vector<int64_t> regs(numRegs, 0);
  std::copy(constPool.begin(), constPool.end(), regs.begin());

  // Pass arguments
  for (auto& pair : varToReg) {
    auto it = args.find(pair.first);
    TORCH_CHECK(
        it != args.end(),
        "Argument ",
        pair.first->name_hint(),
        " is not provided");
    regs[pair.second] = it->second;
  }

  // Execute instructions
  auto idx = constPool.size() + varToReg.size();
  for (auto& inst : insts) {
    int64_t result = 0;
    switch (inst.op) {
#define EXEC_BINARY_OP(type, op)                     \
  case k##type:                                      \
    result = regs[inst.src[0]] op regs[inst.src[1]]; \
    break;

      EXEC_BINARY_OP(Add, +)
      EXEC_BINARY_OP(Sub, -)
      EXEC_BINARY_OP(Mul, *)
      EXEC_BINARY_OP(Div, /)
      EXEC_BINARY_OP(Mod, %)

#undef EXEC_BINARY_OP

      case kMin:
        result = std::min(regs[inst.src[0]], regs[inst.src[1]]);
        break;

      case kMax:
        result = std::max(regs[inst.src[0]], regs[inst.src[1]]);
        break;

      case kCompareSelect: {
        auto lhs = regs[inst.src[0]], rhs = regs[inst.src[1]];
        bool cond = true;
        switch (inst.cmp) {
          case kEQ:
            cond = lhs == rhs;
            break;
          case kGT:
            cond = lhs > rhs;
            break;
          case kGE:
            cond = lhs >= rhs;
            break;
          case kLT:
            cond = lhs < rhs;
            break;
          case kLE:
            cond = lhs <= rhs;
            break;
          case kNE:
            cond = lhs != rhs;
            break;
        }
        result = cond ? regs[inst.src[2]] : regs[inst.src[3]];
        break;
      }

      case kOther:
        result = regs[inst.src[0]] ? regs[inst.src[1]] : regs[inst.src[2]];
        break;

      default:
        TORCH_CHECK(false, "Cannot execution operation ", inst.op);
    }
    regs[idx++] = result;
  }

  return regs.back();
}

static constexpr auto kOpWidth = 14;
static constexpr auto kRegWidth = 4;

static std::string cmpStrs[] = {"=", ">", ">=", "<", "<=", "!="};
static std::unordered_map<IRNodeType, std::string> nodeTypeStrs{
    {kAdd, "Add"},
    {kSub, "Sub"},
    {kMul, "Mul"},
    {kDiv, "Div"},
    {kMod, "Mod"},
    {kMin, "Min"},
    {kMax, "Max"},
    {kCompareSelect, "CompareSelect"},
    {kOther, "IfThenElse"},
};

std::ostream& operator<<(std::ostream& os, const DimInst& inst) {
  TORCH_CHECK(
      nodeTypeStrs.count(inst.op), "Cannot print instruction of op ", inst.op);
  os << std::setw(kOpWidth) << std::setiosflags(std::ios::left)
     << nodeTypeStrs[inst.op] << std::resetiosflags(std::ios::left);
  switch (inst.op) {
    case kAdd:
    case kSub:
    case kMul:
    case kDiv:
    case kMod:
    case kMax:
    case kMin:
      os << std::setw(kRegWidth) << inst.src[0] << std::setw(kRegWidth)
         << inst.src[1];
      break;

    case kCompareSelect:
      os << std::setw(kRegWidth) << inst.src[0] << std::setw(kRegWidth)
         << inst.src[1] << std::setw(kRegWidth) << inst.src[2]
         << std::setw(kRegWidth) << inst.src[3] << std::setw(kRegWidth)
         << cmpStrs[inst.cmp];
      break;

    case kOther:
      os << std::setw(kRegWidth) << inst.src[0] << std::setw(kRegWidth)
         << inst.src[1] << std::setw(kRegWidth) << inst.src[2];
      break;

    default:
      TORCH_CHECK(false, "Unreachable");
  }
  os << '\n';
  return os;
}

void DimExprEvaluator::dump() const {
  // Constants
  for (auto i : c10::irange(constPool.size()))
    std::cout << std::setw(kRegWidth) << i << '\t' << constPool[i] << '\n';
  // Variables
  std::vector<VarPtr> vars(varToReg.size());
  for (auto& pair : varToReg)
    vars[pair.second - constPool.size()] = pair.first;
  uint32_t idx = constPool.size();
  for (auto& var : vars)
    std::cout << std::setw(kRegWidth) << idx++ << '\t' << var->name_hint()
              << '\n';
  // Instructions
  for (auto& inst : insts)
    std::cout << std::setw(kRegWidth) << idx++ << '\t' << inst;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch