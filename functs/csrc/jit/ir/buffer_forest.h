#pragma once

#include <memory>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

struct VarVersion {
  VarVersion(Value *v) : var(v) {}
  Value *var;
  std::vector<Value *> version_stack{};
};

struct BufferNode {
  BufferNode(Value *v) { bufferNode_ = std::make_shared<VarVersion>(v); }
  std::shared_ptr<VarVersion> bufferNode_;
  std::shared_ptr<BufferNode> pointsTo{nullptr};
  std::vector<std::shared_ptr<BufferNode>> pointedFrom{};
};

struct BufferTree {
  std::shared_ptr<BufferNode> root;
  std::vector<Node *> mutations;

  BufferTree(std::shared_ptr<BufferNode> node) : root(node) {}

  void addEdgeToBufferTree(Value *from, Value *to);
  bool find(Value* v);
};

// BufferForest to record alias relationship and version relationship
// As for alias relationship, `%a` `pointsTo` `%b` predict that `%a` is a `view`
// of `%b`, i.e. `%b = aten::select(%a, ...)` As for version relationship, a
// `pointsTo` connects variable before and after functionalization i.e. A
// functionalization converts `%a.0 = aten::copy_(...)` to `%a.1 =
// immut::assign(...)` => `%a.0` `pointsTo` `%a.1`

// Note: you shold construct buffer tree first and then construct update version
// when update version begins, you should not chang alias relationship
// dynamicly!!!

class BufferForest {
public:
  BufferForest() = default;
  void addEdgeToBufferForest(Value *from, Value *to);
  void addMutationToBufferForest(Node *node);
  bool isBufferMutation(Node *node);

  std::shared_ptr<BufferTree> getBufferTreeOrNone(Value *v);

  std::vector<std::shared_ptr<BufferTree>> bufferForest_;
};
} // namespace jit
} // namespace torch
