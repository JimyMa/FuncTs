#include "functs/csrc/jit/ir/alias_analysis.h"
#include "functs/csrc/jit/ir/buffer_forest.h"
#include <ATen/core/interned_strings.h>
#include <c10/util/Optional.h>
#include <functs/csrc/jit/passes/remove_inplace.h>
#include <memory>

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

void RemoveInplaceImpl(Block *b, std::shared_ptr<AliasDbCopy> aliasDb) {
  auto nodes = b->nodes();
  for (auto it = nodes.begin(); it != nodes.end();) {
    auto node = *it;
    for (auto &block : node->blocks()) {
      RemoveInplaceImpl(block, aliasDb);
    }

    // preprocess: convert `aten::compute_` buffer ==> `aten::compute` and
    // `aten::copy`
    if (isMutating(node) && aliasDb->elementMap().count(node->input(0)) &&
        node->input(0)->type()->kind() != TypeKind::ListType &&
        node->input(0)->type()->kind() != TypeKind::DictType &&
        node->input(0)->type()->kind() != TypeKind::ClassType) {
      if (aten::copy_ != node->kind()) {
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
        node->output()->replaceAllUsesWith(node->input(0));
        ++it;
        node->destroy();
      } else {
        node->output()->replaceAllUsesWith(node->input(0));
        ++it;
      }
    } else {
      ++it;
    }
  }
}

void RemoveInplace(std::shared_ptr<Graph> g) {
  auto aliasDb = std::make_shared<AliasDbCopy>(g);
  RemoveInplaceImpl(g->block(), aliasDb);
}

} // namespace jit
} // namespace torch
