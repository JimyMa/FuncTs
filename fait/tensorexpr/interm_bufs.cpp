#include "interm_bufs.h"

#include <torch/csrc/jit/tensorexpr/loopnest.h>

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {

class IntermBufCollector : public IRVisitor {
 public:
  IntermBufCollector(const std::unordered_set<BufPtr> &outBufs)
      : outBufs(outBufs) {}

  auto &&getIntermBufs() && { return std::move(intermBufs); }

  void visit(StorePtr store) override {
    auto buf = store->buf();
    if (outBufs.count(buf)) return;
    if (std::count(intermBufs.begin(), intermBufs.end(), buf)) return;
    intermBufs.push_back(buf);
  }

 private:
  const std::unordered_set<BufPtr> &outBufs;
  std::vector<BufPtr> intermBufs;
};

}  // namespace

std::vector<BufPtr> collectIntermBufs(
    StmtPtr stmt, const std::unordered_set<BufPtr> &outBufs) {
  IntermBufCollector collector(outBufs);
  stmt->accept(&collector);
  return std::move(collector).getIntermBufs();
}

std::vector<StmtPtr> splitAtIntermBufs(ForPtr outerLoop,
                                       const std::vector<BufPtr> &intermBufs,
                                       const std::vector<BufPtr> &resultBufs) {
  // Return single loop if there is no intermediate
  if (intermBufs.empty()) return {outerLoop};

  // Create set of output buffers
  std::unordered_set<BufPtr> outBufs(resultBufs.begin(), resultBufs.end());
  outBufs.insert(intermBufs.begin(), intermBufs.end());

  // Split loop
  auto body = outerLoop->body();
  std::vector<StmtPtr> results;
  for (auto &interm : intermBufs) {
    // Get outer compute loop (blockIdx.x)
    LoopNest nest(outerLoop, outBufs);
    auto loops = nest.getLoopStmtsFor(interm);
    TORCH_CHECK(loops.size() >= 2, "Expect at least 2 nested loops, got ",
                loops.size());
    auto computeLoop = loops[1];

    // Remove compute loop from original block
    TORCH_CHECK(computeLoop->get_parent() == body);
    body->remove_stmt(computeLoop);

    // Create a new outmost loop for this intermediate
    auto newOuter = alloc<For>(outerLoop->var(), outerLoop->start(),
                               outerLoop->stop(), computeLoop);
    newOuter->set_gpu_block_index(1);
    results.push_back(std::move(newOuter));
  }

  results.push_back(outerLoop);
  return results;
}

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch