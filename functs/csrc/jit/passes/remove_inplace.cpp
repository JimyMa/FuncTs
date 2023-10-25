#include <ATen/core/interned_strings.h>
#include <c10/util/Optional.h>
#include <functs/csrc/jit/passes/remove_inplace.h>

namespace torch {
namespace jit {

inline bool isMutating(Node *node, size_t index = 0) {
  auto schema = node->maybeSchema();
  if (!schema)
    return false;
  if (schema->arguments().empty())
    return false;
  return schema->is_mutable({c10::SchemaArgType::input, index});
}

void RemoveInplaceImpl(Block *b) {
  auto nodes = b->nodes();
  for (auto it = nodes.begin(); it != nodes.end();) {
    auto node = *it;
    for (auto &block : node->blocks()) {
      RemoveInplaceImpl(block);
    }

    if (isMutating(node) && aten::copy_ != node->kind()) {
      WithInsertPoint guard(b->param_node()->next());
      auto constant_false = b->owningGraph()->insertConstant(false);

      auto mutSym = node->kind();
      std::string immutOpName(mutSym.toUnqualString());
      immutOpName.pop_back();

      auto immutSym = Symbol::fromQualString(
          std::string(mutSym.ns().toUnqualString()) + "::" + immutOpName);
      auto immutNode = b->owningGraph()
                           ->create(immutSym, node->inputs())
                           ->copyMetadata(node);

      immutNode->insertBefore(node);

      auto copyNode =
          b->owningGraph()->create(aten::copy_, 1)->copyMetadata(node);

      copyNode->addInput(node->input(0));
      copyNode->addInput(immutNode->output());
      copyNode->addInput(constant_false);
      copyNode->insertBefore(node);
      ++it;
      node->destroy();
    } else {
      ++it;
    }
  }
}

void RemoveInplace(std::shared_ptr<Graph> g) { RemoveInplaceImpl(g->block()); }

} // namespace jit
} // namespace torch
