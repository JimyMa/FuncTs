#include "ir_utils.h"

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {

class ReduceChecker : public IRVisitor {
 public:
  ReduceChecker(const VarPtr &iv) : iv(iv) {}

  void visit(ReduceOpPtr reduce) {
    auto &reduceArgs = reduce->reduce_args();
    isReduce |= std::count(reduceArgs.begin(), reduceArgs.end(), iv);
  }

  operator bool() const { return isReduce; }

 private:
  VarPtr iv;
  bool isReduce = false;
};

}  // namespace

bool isReductionLoop(ForPtr loop) {
  ReduceChecker checker(loop->var());
  loop->body()->accept(&checker);
  return checker;
}

namespace {

class StoreBufCollector : public IRVisitor {
 public:
  StoreBufCollector(std::unordered_set<BufPtr> &storedBufs)
      : storedBufs(storedBufs) {}

  void visit(StorePtr store) override { storedBufs.insert(store->buf()); }

 private:
  std::unordered_set<BufPtr> &storedBufs;
};

}  // namespace

std::unordered_set<BufPtr> getStoredBufs(StmtPtr stmt) {
  std::unordered_set<BufPtr> storedBufs;
  StoreBufCollector collector(storedBufs);
  stmt->accept(&collector);
  return std::move(storedBufs);
}

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch