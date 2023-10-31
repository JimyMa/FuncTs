#include "flatten_indices.h"

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {

class IndexFlattener : public IRMutator {
 public:
  ExprPtr mutate(LoadPtr v) override {
    std::vector<ExprPtr> indices;
    for (auto &idx : v->indices()) indices.push_back(idx->accept_mutator(this));
    auto flatIndex = flatten_index(v->buf()->dims(), std::move(indices),
                                   v->buf()->strides());
    v->set_indices({flatIndex});
    return v;
  }

  StmtPtr mutate(StorePtr v) override {
    v->set_value(v->value()->accept_mutator(this));
    if (v->indices().size() == 1) return v;
    v->set_indices(
        {flatten_index(v->buf()->dims(), v->indices(), v->buf()->strides())});
    return v;
  }
};

}  // namespace

StmtPtr flattenIndices(StmtPtr stmt) {
  IndexFlattener mutator;
  return stmt->accept_mutator(&mutator);
}

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch