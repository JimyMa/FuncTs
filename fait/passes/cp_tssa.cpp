#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/runtime/operator.h>

#include "common_passes.h"
#include "tensor_ssa.h"
#include "util/ir.h"
#include "util/traits.h"

namespace torch {
namespace jit {

static void listUnpack(Stack &stack, size_t num_outputs) {
  auto list = pop(stack).toList();
  TORCH_CHECK(list.size() == num_outputs, "Expected ", num_outputs,
              " elements in a list but found ", list.size());
  stack.insert(stack.end(), list.begin(), list.end());
}

static void tupleConstruct(Stack &stack, size_t num_inputs) {
  switch (num_inputs) {
    case 0:
      stack.emplace_back(c10::ivalue::Tuple::create());
      break;
    case 1:
      stack.back() = c10::ivalue::Tuple::create(std::move(stack.back()));
      break;
    case 2: {
      auto tuple =
          c10::ivalue::Tuple::create(std::move(stack[stack.size() - 2]),
                                     std::move(stack[stack.size() - 1]));
      stack.pop_back();
      stack.back() = std::move(tuple);
      break;
    }
    case 3: {
      auto tuple =
          c10::ivalue::Tuple::create(std::move(stack[stack.size() - 3]),
                                     std::move(stack[stack.size() - 2]),
                                     std::move(stack[stack.size() - 1]));
      stack.pop_back();
      stack.pop_back();
      stack.back() = std::move(tuple);
      break;
    }
    default: {
      std::vector<IValue> elems{
          std::make_move_iterator(stack.end() - num_inputs),
          std::make_move_iterator(stack.end())};
      drop(stack, num_inputs - 1);
      stack.back() = c10::ivalue::Tuple::create(std::move(elems));
      break;
    }
  }
}

static void listConstruct(Stack &stack, const c10::Type &list_type,
                          size_t num_inputs) {
  auto makeList = [](Stack &stack, const c10::Type &list_type,
                     size_t num_inputs) {
    c10::List<IValue> vals(list_type.containedType(0));
    vals.reserve(num_inputs);
    for (size_t i = stack.size() - num_inputs; i < stack.size(); ++i) {
      vals.push_back(std::move(stack[i]));
    }
    drop(stack, num_inputs);
    return vals;
  };
  stack.emplace_back(makeList(stack, list_type, num_inputs));
}

static void ifCond(Stack &stack, size_t num_inputs) {}

static c10::optional<Stack> tryRunNodes(Node *node) {
  // Skip if no outputs are produced
  if (node->outputs().empty()) return c10::nullopt;

  // Do not run on mutating nodes
  if (isMutating(node)) return c10::nullopt;

  // Do not run on aliasing nodes whose output is assigned/updated
  if (isAliasing(node)) {
    auto output = node->output(0);
    if (std::any_of(output->uses().begin(), output->uses().end(),
                    [](const Use &use) {
                      auto kind = use.user->kind();
                      return kind == tssa::Assign || kind == tssa::Update;
                    }))
      return c10::nullopt;
  }

  // Push inputs to stack
  Stack stack;
  for (auto input : node->inputs()) {
    if (auto ival = toIValue(input))
      stack.push_back(*ival);
    else
      return c10::nullopt;
  }

  // Run nodes for different symbols
  switch (node->kind()) {
    case prim::ListUnpack: {
      if (stack.back().toList().size() != node->outputs().size())
        return c10::nullopt;
      listUnpack(stack, node->outputs().size());
    } break;

    case prim::TupleConstruct: {
      tupleConstruct(stack, node->inputs().size());
    } break;

    case prim::ListConstruct: {
      listConstruct(stack, node->output()->type()->expectRef<ListType>(),
                    node->inputs().size());
    } break;

    default: {
      auto schema = node->maybeSchema();
      if (!schema || schema->is_vararg()) return c10::nullopt;
      try {
        node->getOperation()(stack);
      } catch (...) {
        return c10::nullopt;
      }
    } break;
  }

  return stack;
}

Node *foldIfCond(Node *ifNode, bool &changed) {
  // Check if the condition can be folded
  IfView view(ifNode);
  auto constCond = constant_as<bool>(view.cond());
  if (!constCond.has_value()) return nullptr;

  // Move nodes in corresponding block out of node
  auto block = constCond.value() ? view.thenBlock() : view.elseBlock();
  auto graph = ifNode->owningGraph();
  graph->setInsertPoint(ifNode->next());
  std::unordered_map<Value *, Value *> valueMap;
  cloneNodesTo(block->nodes().front(), block->nodes().back(), ifNode, valueMap);

  // Replace outputs
  auto blockOuts = block->outputs();
  for (auto i : c10::irange(blockOuts.size())) {
    auto output = blockOuts[i];
    ifNode->output(i)->replaceAllUsesWith(
        valueMap.count(output) ? valueMap[output] : output);
  }

  changed = true;
  return remove(ifNode);
}

bool FoldConstantsTSSA(const std::shared_ptr<Graph> &graph) {
  bool changed = false;
  rewrite(graph->block(), [&](Node *node) -> Node * {
    // Handle `if` node
    if (node->kind() == prim::If) return foldIfCond(node, changed);

    // Check if its inputs and outputs are not mutated
    if (std::any_of(node->inputs().begin(), node->inputs().end(), isMutated))
      return nullptr;
    if (std::any_of(node->outputs().begin(), node->outputs().end(), isMutated))
      return nullptr;

    // Try run the nodes
    auto outputs = tryRunNodes(node);
    if (!outputs) return nullptr;

    // Insert constant nodes to the graph
    graph->setInsertPoint(node);
    for (auto i = 0u; i < (*outputs).size(); i++) {
      auto ival = (*outputs)[i];
      auto cnstVal = tryInsertConstant(*graph, ival);
      if (!cnstVal) continue;
      if (ival.isNone()) (*cnstVal)->setType(node->outputs()[i]->type());
      node->outputs()[i]->replaceAllUsesWith(*cnstVal);
      changed = true;
    }

    return nullptr;
  });
  return changed;
}

}  // namespace jit
}  // namespace torch
