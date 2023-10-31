#include "extract_common_cond.h"

#include <torch/csrc/jit/tensorexpr/ir_cloner.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {

using CondCountMap =
    std::unordered_map<CompareSelectPtr, size_t, ExprHasher, ExprEq>;

class CondCounter : public IRVisitor {
 public:
  auto &&getCondCount() && { return std::move(condCount); }

  void visit(CompareSelectPtr cmp) override {
    IRVisitor::visit(cmp);

    // Check if the expression is in the form `expr cmp c ? 1 : 0`
    if (!cmp->rhs()->isConstant()) return;
    auto val1 = to<IntImm>(cmp->ret_val1());
    if (!val1 || val1->value() != 1) return;
    auto val2 = to<IntImm>(cmp->ret_val2());
    if (!val2 || val2->value() != 0) return;

    // Count the expression
    auto it = condCount.find(cmp);
    if (it != condCount.end())
      it->second++;
    else
      condCount.insert({cmp, 1});
  }

 private:
  CondCountMap condCount;
};

using CondValueMap =
    std::unordered_map<CompareSelectPtr, int, ExprHasher, ExprEq>;

class CondRewriter : public IRCloner {
 public:
  CondRewriter(const CondValueMap &condValues) : condValues(condValues) {}

  ExprPtr mutate(CompareSelectPtr cmp) override {
    auto it = condValues.find(cmp);
    if (it != condValues.end())
      return alloc<IntImm>(it->second);
    else
      return IRCloner::mutate(cmp);
  }

 private:
  const CondValueMap &condValues;
};

}  // namespace

static ExprPtr createExtractedConds(
    const ExprPtr &expr, const std::vector<CompareSelectPtr> &conds, size_t idx,
    std::vector<std::pair<CompareSelectPtr, int>> &assumed) {
  if (idx == conds.size()) {
    CondValueMap condValues(assumed.begin(), assumed.end());
    CondRewriter rewriter(condValues);
    auto lastExpr = expr;
    while (true) {
      auto newExpr =
          IRSimplifier::simplify(lastExpr->accept_mutator(&rewriter));
      if (ExprEq()(newExpr, lastExpr)) break;
      lastExpr = newExpr;
    }
    return lastExpr;
  } else {
    auto &cond = conds[idx];
    assumed.push_back({cond, 1});
    auto trueBr = createExtractedConds(expr, conds, idx + 1, assumed);
    assumed.back().second = 0;
    auto falseBr = createExtractedConds(expr, conds, idx + 1, assumed);
    assumed.pop_back();
    return alloc<IfThenElse>(cond, trueBr, falseBr);
  }
}

ExprPtr extractCommonCond(ExprPtr expr) {
  // Count all conditions in the expression
  CondCounter counter;
  expr->accept(&counter);
  auto condCount = std::move(counter).getCondCount();
  if (condCount.empty()) return expr;

  // Choose frequent conditions whose extraction is beneficial
  std::vector<std::pair<CompareSelectPtr, size_t>> condCountVec(
      condCount.begin(), condCount.end());
  std::sort(condCountVec.begin(), condCountVec.end(),
            [](auto &lhs, auto &rhs) { return lhs.second > rhs.second; });
  if (condCountVec.front().second == 1) return expr;
  std::vector<CompareSelectPtr> conds;
  for (auto i : c10::irange(condCountVec.size())) {
    auto &pair = condCountVec[i];
    if (pair.second <= (1 << i)) break;
    conds.push_back(pair.first);
  }

  // Construct `IfThenElse` expressions
  std::vector<std::pair<CompareSelectPtr, int>> assumed;
  return createExtractedConds(expr, conds, 0, assumed);
}

namespace {

class ForStopRefactor : public IRMutator {
 public:
  StmtPtr mutate(ForPtr loop) override {
    loop->set_stop(extractCommonCond(loop->stop()));
    loop->set_body(loop->body()->accept_mutator(this));
    return loop;
  }
};

}  // namespace

StmtPtr refactorForStop(StmtPtr stmt) {
  ForStopRefactor refactor;
  return stmt->accept_mutator(&refactor);
}

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch