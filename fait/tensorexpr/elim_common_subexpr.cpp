#include "elim_common_subexpr.h"

#include "util/name.h"

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {

using Binding = std::pair<VarPtr, ExprPtr>;

class VarCreator : public IRVisitor {
 public:
  VarCreator(NameGenerator &nameGen) : nameGen(nameGen) {}

  auto &getExprVarMap() const { return exprToVar; }

#define CREATE_VAR_FOR_BINARY(op)     \
  void visit(op##Ptr expr) override { \
    createVarFor(expr->lhs());        \
    createVarFor(expr->rhs());        \
  }

  CREATE_VAR_FOR_BINARY(Add)
  CREATE_VAR_FOR_BINARY(Sub)
  CREATE_VAR_FOR_BINARY(Mul)
  CREATE_VAR_FOR_BINARY(Div)
  CREATE_VAR_FOR_BINARY(Mod)
  CREATE_VAR_FOR_BINARY(Max)
  CREATE_VAR_FOR_BINARY(Min)

#undef CREATE_VAR_FOR_BINARY

  void visit(IntrinsicsPtr intrin) override {
    for (auto &param : intrin->params()) createVarFor(param);
  }

  void visit(LoadPtr load) override {
    for (auto &index : load->indices()) createVarFor(index);
  }

  void visit(CompareSelectPtr cmpSel) override {
    createVarFor(cmpSel->lhs());
    createVarFor(cmpSel->rhs());
    createVarFor(cmpSel->ret_val1());
    createVarFor(cmpSel->ret_val2());
  }

  void visit(IfThenElsePtr ite) override {
    createVarFor(ite->condition());
    createVarFor(ite->true_value());
    createVarFor(ite->false_value());
  }

 private:
  void createVarFor(ExprPtr expr) {
    // Skip constants and variables
    if (expr->isConstant() || to<Var>(expr)) return;

    // Find the variable if the expression is numbered before
    auto idIter = exprValueIds.find(expr);
    if (idIter != exprValueIds.end()) {
      exprToVar.insert({expr, idIter->second});
      return;
    }

    // Visit subexpressions
    expr->accept(this);

    // Create a new variable for the expression
    auto var = Var::make(nameGen.generate(), expr->dtype()).AsNode<Var>();
    exprValueIds.insert({expr, var});
    exprToVar.insert({expr, var});
  }

  std::unordered_map<ExprPtr, VarPtr, ExprHasher, ExprEq> exprValueIds;
  std::unordered_map<ExprPtr, VarPtr> exprToVar;
  NameGenerator &nameGen;
};

class ExprReplacer : public IRMutator {
 public:
  ExprReplacer(const std::unordered_map<ExprPtr, VarPtr> &exprToVar)
      : exprToVar(exprToVar) {}

  auto &&getBindings() && { return std::move(bindings); }

#define REPLACE_EXPR(node)                   \
  ExprPtr mutate(node##Ptr expr) override {  \
    return replace(IRMutator::mutate(expr)); \
  }

  REPLACE_EXPR(Cast)
  REPLACE_EXPR(Add)
  REPLACE_EXPR(Sub)
  REPLACE_EXPR(Mul)
  REPLACE_EXPR(Div)
  REPLACE_EXPR(Mod)
  REPLACE_EXPR(Max)
  REPLACE_EXPR(Min)
  REPLACE_EXPR(Intrinsics)
  REPLACE_EXPR(Load)
  REPLACE_EXPR(CompareSelect)
  REPLACE_EXPR(IfThenElse)

 private:
  ExprPtr replace(ExprPtr expr) {
    if (!exprToVar.count(expr)) return expr;
    auto var = exprToVar.at(expr);
    if (createdVars.count(var)) return var;
    bindings.push_back({var, expr});
    createdVars.insert(var);
    return var;
  }

  const std::unordered_map<ExprPtr, VarPtr> &exprToVar;
  std::unordered_set<VarPtr> createdVars;
  std::vector<Binding> bindings;
};

struct LetInsertTask {
  StmtPtr stmt;
  std::vector<Binding> bindings;
};

class LetInserter : public IRMutator {
 public:
  LetInserter() : nameGen("_var_") {}

  StmtPtr mutate(CondPtr cond) override {
    VarCreator creator(nameGen);
    ExprReplacer replacer(creator.getExprVarMap());
    cond->set_condition(
        processExpr(cond->condition(), cond, creator, replacer));
    cond->set_true_stmt(cond->true_stmt()->accept_mutator(this));
    if (cond->false_stmt())
      cond->set_false_stmt(cond->false_stmt()->accept_mutator(this));
    produceTask(cond, std::move(replacer).getBindings());
    return cond;
  }

  StmtPtr mutate(StorePtr store) override {
    VarCreator creator(nameGen);
    ExprReplacer replacer(creator.getExprVarMap());
    std::vector<ExprPtr> indices;
    for (auto &index : store->indices())
      indices.push_back(processExpr(index, store, creator, replacer));
    store->set_indices(indices);
    store->set_value(processExpr(store->value(), store, creator, replacer));
    produceTask(store, std::move(replacer).getBindings());
    return store;
  }

  StmtPtr mutate(ForPtr loop) override {
    VarCreator creator(nameGen);
    ExprReplacer replacer(creator.getExprVarMap());
    loop->set_stop(processExpr(loop->stop(), loop, creator, replacer));
    loop->set_body(loop->body()->accept_mutator(this));
    produceTask(loop, std::move(replacer).getBindings());
    return loop;
  }

  StmtPtr mutate(BlockPtr block) override {
    // Mutate inner statements
    IRMutator::mutate(block);

    // Perform insertion tasks
    if (!insertTasks.count(block)) return block;
    auto stmtList = block->stmts();
    std::vector<StmtPtr> stmts(stmtList.begin(), stmtList.end());

    for (auto &task : insertTasks[block]) {
      // Find insertion position
      auto iter = std::find(stmts.begin(), stmts.end(), task.stmt);
      TORCH_CHECK(iter != stmts.end(), "Cannot find statement in block");

      // Create and insert bindings
      std::vector<StmtPtr> lets;
      for (auto &binding : task.bindings)
        lets.push_back(
            Let::make(VarHandle(binding.first), ExprHandle(binding.second)));
      stmts.insert(iter, lets.begin(), lets.end());
    }

    block->set_stmts(stmts);

    // Remove tasks for this block
    insertTasks.erase(block);

    return block;
  }

 private:
  ExprPtr processExpr(ExprPtr expr, const StmtPtr &stmt, VarCreator &creator,
                      ExprReplacer &replacer) {
    auto block = to<Block>(stmt->get_parent());
    if (!block) return expr;
    expr->accept(&creator);
    return expr->accept_mutator(&replacer);
  }

  void produceTask(StmtPtr stmt, std::vector<Binding> &&bindings) {
    if (bindings.empty()) return;
    auto block = to<Block>(stmt->get_parent());
    TORCH_CHECK(block, "Cannot insert let bindings to non-block statement");
    LetInsertTask task{std::move(stmt), std::move(bindings)};
    if (insertTasks.count(block)) {
      insertTasks[block].push_back(std::move(task));
    } else {
      insertTasks.insert({block, {std::move(task)}});
    }
  }

  std::unordered_map<BlockPtr, std::vector<LetInsertTask>> insertTasks;
  NameGenerator nameGen;
};

}  // namespace

StmtPtr eliminateCommonSubexpr(StmtPtr stmt) {
  LetInserter inserter;
  stmt->accept_mutator(&inserter);
  return stmt;
}

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch