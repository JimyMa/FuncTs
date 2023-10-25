#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <set>

#include <ATen/core/symbol.h>
#include <c10/util/Exception.h>
#include <functs/csrc/jit/ir/alias_analysis.h>
#include <functs/csrc/jit/ir/symbol_ext.h>
#include <functs/csrc/jit/passes/convert_to_tensorssa.h>
#include <functs/csrc/jit/passes/remove_inplace.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/utils/memory_dag.h>
// #include <torch/csrc/jit/passes/utils/memory_dag.h>

using namespace c10;

namespace torch {
namespace jit {

void GetBufferTree(std::shared_ptr<Graph> g, AliasDbCopy &aliasDb_buffer_tree) {
  // Note: Unsupported alias relationship by now:
  // // - An element points to more than one elements (i.e. a <- c, b <- c)
  // TODO: Cannot support add Container alias analysis by now
  // Note: Only intra-procedure are supported by now

  // g: Graph-level IR based tensor program
  // alias_db_origin: origin alias relationship
  // alias_db_buffer_tree: modified alias relationship

  // Step 1. visit all elements in aliasdb, delete all unsupported alias
  // relationship.
  // // 1. find all ambigious alias relationship
  // // 2. delete them from aliasDb_buffer_tree

  bool any_changed = true;
  auto elementMap = aliasDb_buffer_tree.elementMapMutable();

  ska::flat_hash_map<const Value *, Element *> ambigious_alias;

  // Step 1

  for (const auto &ptrPair : elementMap) {
    auto value = ptrPair.first;
    auto element = ptrPair.second;

    if (element->pointsTo.count() > 1 /* Step 1.1: count > 1 */ ||
        aliasDb_buffer_tree.mayAliasWildcard(value) /* wildcard node */) {
      ambigious_alias[ptrPair.first] = ptrPair.second;
    }
  }

  ska::flat_hash_map<const Value *, Element *> may_alias;

  // Step 2
  // get may_alias
  for (const auto &ptrPair : elementMap) {
    for (const auto &ptrPairAmbigious : ambigious_alias) {
      if (aliasDb_buffer_tree.mayAlias(ptrPair.first, ptrPairAmbigious.first) &&
          !may_alias.count(ptrPair.first)) {
        may_alias.insert(ptrPair);
      }
    }
  }

  // Delete from elementMap
  for (const auto &ptrPair : may_alias) {
    aliasDb_buffer_tree.elementMapErase(ptrPair.first);
  }
}

void TensorSSAAliasRemoval(Block *b, AliasDbCopy *buffer_forest) {
  auto nodes = b->nodes();
  for (auto node = nodes.front(); node != nodes.back();) {
    auto blocks = node->blocks();
    for (auto block : blocks) {
      TensorSSAAliasRemoval(block, buffer_forest);
    }

    // Judge if a node is a write node, if true, convert to immutable equivalent
    // Step 1. Pass Up
    // Unsupported has been eliminated from alias set
    auto elementMap = buffer_forest->elementMapMutable();
    if (aten::copy_ == node->kind() && elementMap.count(node->output())) {
      // TODO: only tensor values are considered...
      const Value *points_to;
      Value *leaf_value = node->output(0);
      Value *pass_up_value = node->input(0);
      b->owningGraph()->setInsertPoint(node->next());
      do {
        auto points_to_elem =
            buffer_forest->fromIndex(*elementMap[leaf_value]->pointsTo.begin());

        if (points_to_elem->values.size() > 0) {
          points_to = *(points_to_elem->values.begin());
          Node *pass_up_node;
          auto leaf_node = leaf_value->node();
          if (aten::select == leaf_node->kind() ||
              immutable::Select == leaf_node->kind()) {
            pass_up_node = b->owningGraph()->create(
                immutable::SelectReverse, leaf_value->node()->inputs(), 1);
            pass_up_node->insertInput(1, pass_up_value);
          } else if (aten::copy_ == leaf_node->kind() ||
                     immutable::Assign == leaf_node->kind()) {
            pass_up_node = b->owningGraph()->create(
                immutable::Assign, leaf_value->node()->inputs(), 1);
          } else {
            AT_ASSERT(false, "Unknown alias operator when pass up",
                      leaf_node->kind().toQualString());
          }
          pass_up_node->output(0)->setType(leaf_value->type());
          pass_up_value = pass_up_node->output(0);
          b->owningGraph()->insertNode(pass_up_node);
          b->owningGraph()->setInsertPoint(pass_up_node->next());
          leaf_value = const_cast<Value *>(points_to);
        } else {
          points_to = nullptr;
        }
      } while (points_to);

      // Step 2. pass down
      // reconstruct elementMap_
      Value *pass_down_value = pass_up_value;
      Value *root_value = leaf_value;

      buffer_forest->createValueByCopy(pass_down_value, root_value);
      buffer_forest->destroyValue(root_value);

      root_value->replaceAllUsesAfterNodeWith(pass_up_value->node(),
                                              pass_down_value);

      b->owningGraph()->setInsertPoint(pass_up_value->node()->next());

      std::function<void()> pass_down;
      pass_down = [&]() -> void {
        auto elementMap = buffer_forest->elementMapMutable();
        for (auto pointedFromIndex : elementMap[pass_down_value]->pointedFrom) {
          auto from_elem = buffer_forest->fromIndex(pointedFromIndex);

          auto from_value = const_cast<Value *>(*from_elem->values.begin());
          auto from_node = from_value->node();

          if (!(from_node->isAfter(node))) {
            Node *pass_down_node;
            if (immutable::Select == from_node->kind()) {
              pass_down_node = b->owningGraph()->create(
                  immutable::Select, const_cast<Node *>(from_node)->inputs(),
                  1);
              pass_down_node->replaceInput(0, pass_down_value);
            } else if (aten::copy_ == node->kind()) {
              pass_down_node = b->owningGraph()->create(
                  immutable::Assign, const_cast<Node *>(from_node)->inputs(),
                  1);
              pass_down_node->replaceInput(0, pass_down_value);
              pass_down_node->replaceInput(1, pass_down_value);
            } else {
              AT_ASSERT(false, "Unknown alias operator when pass down",
                        from_node->kind().toQualString());
            }

            b->owningGraph()->insertNode(pass_down_node);
            b->owningGraph()->setInsertPoint(pass_down_node->next());
            pass_down_value = pass_down_node->output(0);
            pass_down_value->setType(
                const_cast<Node *>(from_node)->output(0)->type());
            buffer_forest->createValueByCopy(pass_down_value, from_value);
            buffer_forest->destroyValue(from_value);
            from_node->output(0)->replaceAllUsesAfterNodeWith(pass_down_node,
                                                              pass_down_value);
            root_value = from_node->output(0);
            pass_down();
          }
        }
      };
      pass_down();

      node->destroy();
      node = b->owningGraph()->insertPoint()->next();
    } else {
      node = node->next();
    }
  }
}

void TensorSSAPropagation(std::shared_ptr<Graph> graph);
void TensorSSARename(std::shared_ptr<Graph> graph);

void TensorSSAImmutablize(Block *b, AliasDbCopy *buffer_forest) {
  auto nodes = b->nodes();
  for (auto node = nodes.front(); node != nodes.back();) {
    auto blocks = node->blocks();
    for (auto block : blocks) {
      TensorSSAImmutablize(block, buffer_forest);
    }
    auto elementMap = buffer_forest->elementMap();
    if (elementMap.count(node->output(0))) {
      if (aten::select == node->kind()) {
        auto immutableNode =
            b->owningGraph()->create(immutable::Select, node->inputs(), 1);
        immutableNode->output()->setType(node->output()->type());
        immutableNode->insertAfter(node);
        node->output()->replaceAllUsesWith(immutableNode->output());
        node->destroy();
        node = immutableNode->next();
      } else {
        node = node->next();
      }
    } else {
      node = node->next();
    }
  }
}

void TensorSSABufferTreeViewImmutablize(Block *b, AliasDbCopy *buffer_forest) {
  // traversal aliasDb buffer forest, convert aten::view to immut::select
  // and change the alias relationship at the same time.
  auto elementMap = buffer_forest->elementMap();
  bool anyChanged = true;

  auto get_all_buffer_nodes = [&]() -> std::vector<Node *> {
    std::vector<Node *> result;
    for (auto elementMapPtr : elementMap) {
      auto from_value = elementMapPtr.first;
      if (aten::select == from_value->node()->kind()) {
        result.push_back(const_cast<Value *>(from_value)->node());
      }
    }
    return result;
  };
  auto all_buffer_nodes = get_all_buffer_nodes();
  for (auto &node : all_buffer_nodes) {
    // add a new immutable operator
    auto new_node =
        b->owningGraph()->create(immutable::Select, node->inputs(), 1);

    new_node->output()->setType(node->output()->type());
    new_node->insertAfter(node);
    node->output()->replaceAllUsesWith(new_node->output());

    // change alias relationship
    buffer_forest->createValueByCopy(new_node->output(), node->output());
    buffer_forest->destroyValue(const_cast<Value *>(node->output()));

    node->destroy();
  }
}

void DumbRemoveInterPrecedureMutation(std::shared_ptr<Graph> graph) {
  for (auto input : graph->inputs()) {
    WithInsertPoint p(graph->block()->param_node()->next());
    if (input->type()->cast<TensorType>()) {
      auto copy_node =
          graph
              ->create(aten::clone, {input, graph->insertConstant(IValue())}, 1)
              ->insertAfter(graph->block()->param_node()->next());
      copy_node->output()->copyMetadata(input);
      input->replaceAllUsesWith(copy_node->output());
    }
  }
}

void ConvertToTensorSSA(std::shared_ptr<Graph> graph) {
  std::cout << "Origin Graph: " << std::endl;
  graph->dump();

  // Preprocess: A dumb pass to eliminate interprecedure view
  std::cout << "dumb remove inter precedure mutation begin..." << std::endl;
  DumbRemoveInterPrecedureMutation(graph);
  std::cout << "dumb remove inter precedure mutation end..." << std::endl;

  // Step 0. convert inplace operator (add_, mul_, ...) to copy
  std::cout << "remove inplace begin..." << std::endl;
  RemoveInplace(graph);
  std::cout << "remove inplace end..." << std::endl;

  // AliasDb
  auto aliasDb_origin = AliasDbCopy(graph);
  auto aliasDb_buffer_tree = AliasDbCopy(graph);
  aliasDb_origin.dump();

  graph->dump();
  aliasDb_buffer_tree.dump();

  // Step 1. Get Buffer Tree
  // LOG(INFO) << "Step 1. Get Buffer Tree" << std::endl;
  std::cout << "get Buffer Tree begin..." << std::endl;
  GetBufferTree(graph, aliasDb_buffer_tree);
  std::cout << "get Buffer Tree end..." << std::endl;

  aliasDb_buffer_tree.dump();
  aliasDb_buffer_tree.dumpToGraphvizFile("buffer_tree.dot");

  // Step 2. Regularization `aten::view`, `aten::copy_` to
  // `immut::access`, `immut::assign`
  TensorSSABufferTreeViewImmutablize(graph->block(), &aliasDb_buffer_tree);
  // graph->dump();

  // Step 2. Convert to TensorSSA
  // LOG(INFO) << "Step 2. Functionaliazation" << std::endl;
  TensorSSAAliasRemoval(graph->block(), &aliasDb_buffer_tree);

  graph->dump();
}

} // namespace jit
} // namespace torch
