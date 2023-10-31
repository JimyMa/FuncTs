#include "parallelize_loops.h"

#include "common_passes.h"
#include "type_utils.h"
#include "unroll_loops.h"
#include "util/ir.h"
#include "util/traits.h"

namespace torch {
namespace jit {

void markNonParLoops(Block *block, Node *loop,
                     std::unordered_set<Node *> &nonParLoops) {
  for (auto node : block->nodes()) {
    // Reject loop with value dependency
    auto isLoop = node->kind() == prim::Loop;
    if (isLoop && node->inputs().size() > 2) nonParLoops.insert(node);

    // Recursively visit blocks
    for (auto nested : node->blocks())
      markNonParLoops(nested, isLoop ? node : loop, nonParLoops);

    // Check list operation
    if (node->inputs().empty()) continue;
    auto list = node->input(0);
    if (!list->type()->cast<ListType>()) continue;

    // No need to care about local lists
    auto listDefBlock = list->node()->owningBlock();
    if (listDefBlock == block) continue;

    // Check if the read (`aten::__getitem__`) is supported
    if (!loop) continue;
    auto body = loop->blocks()[0];
    if (node->kind() == aten::__getitem__) {
      auto iterVar = body->inputs()[0];
      auto index = node->input(1);
      if (index->node()->kind() != prim::Constant && index != iterVar) {
        nonParLoops.insert(loop);
        continue;
      }
    }

    // Check if there is only one `aten::append` to the list
    auto numAppend = 0u;
    for (auto &use : list->uses()) {
      auto user = use.user;
      if (!isMutating(user)) continue;
      if (user->kind() != aten::append) {
        nonParLoops.insert(loop);
        continue;
      }
      numAppend++;
    }
    if (numAppend > 1) nonParLoops.insert(loop);
  }
}

void convertLoopToMap(Node *loop, Graph *graph) {
  // Collect input and output lists of the loop
  std::vector<Value *> inLists, outLists;
  std::vector<Value *> inElems, outElems;
  auto body = loop->blocks()[0];
  auto iterVar = body->inputs()[0];
  for (auto node : body->nodes()) {
    switch (node->kind()) {
      case aten::__getitem__: {
        if (node->input(1) != iterVar) continue;
        inLists.push_back(node->input(0));
        inElems.push_back(node->output(0));
        break;
      }

      case aten::append: {
        outLists.push_back(node->input(0));
        outElems.push_back(node->input(1));
        break;
      }
    }
  }

  // Create map node
  auto mapArgs = std::move(inLists);
  mapArgs.insert(mapArgs.begin(), loop->input(0));
  auto mapNode = graph->create(prim::ParallelMap, mapArgs, outLists.size());
  mapNode->setSourceRange(loop->sourceRange());
  mapNode->insertAfter(loop);
  auto mapBlock = mapNode->addBlock();
  mapBlock->cloneFrom(body, [](Value *v) { return v; });
  auto mapIdx = mapBlock->inputs()[0];

  // Process nodes in the body and remap values
  for (auto iter = mapBlock->nodes().begin(); iter != mapBlock->nodes().end();
       ++iter) {
    auto node = *iter;
    switch (node->kind()) {
      case aten::__getitem__: {
        if (node->input(1) != mapIdx) continue;
        auto param = mapBlock->addInput();
        param->setType(node->output(0)->type());
        node->output(0)->replaceAllUsesWith(param);
        iter.destroyCurrent();
        break;
      }

      case aten::append: {
        auto ret = node->input(1);
        mapBlock->insertOutput(mapBlock->outputs().size(), ret);
        iter.destroyCurrent();
        break;
      }
    }
  }
  mapBlock->eraseOutput(0);

  // Remap output values of the map node
  for (auto i = 0u; i < outLists.size(); i++) {
    auto mapOut = mapNode->output(i), loopOut = outLists[i];
    mapOut->setType(loopOut->type());
    loopOut->replaceAllUsesAfterNodeWith(loop, mapOut);
  }

  // Remove loop node and output list definitions
  loop->destroy();
  for (auto outList : outLists) outList->node()->destroy();
}

void ParallelizeLoops(const std::shared_ptr<Graph> &graph) {
  // Eliminate redundant computation
  HoistLoopInvariants(graph);
  EliminateCommonSubexprTSSA(graph);

  // Find parallelizable loops
  std::unordered_set<Node *> nonParLoops;
  markNonParLoops(graph->block(), nullptr, nonParLoops);
  std::vector<Node *> loops;
  traversePostOrder(graph->block(), [&](Node *node) {
    if (node->kind() == prim::Loop && !nonParLoops.count(node))
      loops.push_back(node);
    return true;
  });

  // Convert to parallel maps
  for (auto loop : loops) convertLoopToMap(loop, graph.get());
}

static c10::optional<size_t> getParMapTripCount(Node *parMap) {
  auto len = constant_as<int64_t>(parMap->input(0));
  return mapOpt<size_t>(len, [](int64_t len) { return size_t(len); });
}

static Node *splitAt(Node *prevParMap, Node *splitNode, Graph *graph,
                     ValueTypeMap &refinedTypes) {
  // Find straight returns and dependent values of previous map
  std::unordered_set<Value *> prevStraightRets;
  std::vector<Value *> nextDepPrevs;
  auto prevBlock = prevParMap->blocks().front();
  for (auto node = prevBlock->nodes().front(); node != splitNode;
       node = node->next()) {
    for (auto output : node->outputs()) {
      for (auto &use : output->uses()) {
        auto user = use.user;
        if (user == prevBlock->return_node())
          prevStraightRets.insert(output);
        else if (user == splitNode || user->isAfter(splitNode)) {
          if (std::find(nextDepPrevs.begin(), nextDepPrevs.end(), output) ==
              nextDepPrevs.end())
            nextDepPrevs.push_back(output);
        }
      }
    }
  }

  // Create new parallel map node
  auto numMapOut = prevParMap->outputs().size();
  auto nextParMap = graph->create(prim::ParallelMap, {}, 0);
  nextParMap->insertAfter(prevParMap);
  auto nextBlock = nextParMap->addBlock();

  // Add node inputs and block parameters of the first map to the next map
  std::unordered_map<Value *, Value *> valueMap;
  for (auto i = 0u; i < prevParMap->inputs().size(); i++) {
    auto prevIn = prevParMap->input(i), prevParam = prevBlock->inputs()[i];
    nextParMap->addInput(prevIn);
    auto nextParam = nextBlock->addInput()->setType(prevParam->type());
    transferRefinedType(prevParam, nextParam, refinedTypes);
    valueMap.insert({prevParam, nextParam});
  }

  // Possibly move node outputs and block returns of previous map
  std::vector<Value *> nextRets;
  for (auto i = 0u; i < prevBlock->outputs().size();) {
    auto prevRet = prevBlock->outputs()[i], prevOut = prevParMap->output(i);
    if (!prevStraightRets.count(prevRet)) {
      // move to next map
      nextRets.push_back(prevRet);
      auto nextOut = nextParMap->addOutput()->setType(prevOut->type());
      transferRefinedType(prevOut, nextOut, refinedTypes);
      prevOut->replaceAllUsesWith(nextOut);
      prevBlock->eraseOutput(i);
      prevParMap->eraseOutput(i);
    } else {
      // add to input of next map if it is used by it
      if (std::find(nextDepPrevs.begin(), nextDepPrevs.end(), prevRet) !=
          nextDepPrevs.end()) {
        nextParMap->addInput(prevOut);
        auto nextParam = nextBlock->addInput()->setType(prevRet->type());
        transferRefinedType(prevRet, nextParam, refinedTypes);
        valueMap.insert({prevRet, nextParam});
      }
      i++;  // keep return and output
    }
  }

  // Add dependencies between previous and next maps
  for (auto dep : nextDepPrevs) {
    if (prevStraightRets.count(dep)) continue;
    prevBlock->registerOutput(dep);
    auto prevOut =
        prevParMap->addOutput()->setType(ListType::create(dep->type()));
    auto refinedListTy = createRefinedListType(
        getRefinedType(dep, refinedTypes), getParMapTripCount(prevParMap));
    setRefinedType(prevOut, refinedListTy, refinedTypes);
    nextParMap->addInput(prevOut);
    auto nextParam = nextBlock->addInput()->setType(dep->type());
    transferRefinedType(dep, nextParam, refinedTypes);
    valueMap.insert({dep, nextParam});
  }

  // Move nodes beginning from the split point to the new map
  moveNodesToBlock(splitNode, prevBlock->return_node(), nextBlock, valueMap,
                   &refinedTypes);

  // Add return values to next block
  for (auto ret : nextRets) nextBlock->registerOutput(valueMap.at(ret));

  return nextParMap;
}

static void splitParallelMap(Node *curParMap, Graph *graph,
                             ValueTypeMap &refinedTypes) {
  // Find fusion group and split parallel map
  auto mapBlock = curParMap->blocks().front();
  for (auto node = mapBlock->nodes().front(); node != mapBlock->return_node();
       node = node->next()) {
    // Check if the node is a fusion group
    if (node->kind() != prim::FusionGroup) continue;

    // Split before the group
    if (node->prev()->kind() != prim::Param) {
      curParMap = splitAt(curParMap, node, graph, refinedTypes);
      mapBlock = curParMap->blocks().front();
      node = mapBlock->param_node()->next();
    }

    // Split after the group
    if (node->next()->kind() != prim::Return) {
      curParMap = splitAt(curParMap, node->next(), graph, refinedTypes);
      mapBlock = curParMap->blocks().front();
      node = mapBlock->param_node();
    }
  }
}

void SplitParallelMaps(const std::shared_ptr<Graph> &graph,
                       ValueTypeMap &refinedTypes) {
  // Find all parallel maps
  std::vector<Node *> parMaps;
  traversePostOrder(graph->block(), [&](Node *node) {
    if (node->kind() == prim::ParallelMap) parMaps.push_back(node);
    return true;
  });

  // Split parallel maps for fusion groups
  for (auto parMap : parMaps)
    splitParallelMap(parMap, graph.get(), refinedTypes);

  // Remove unused map inputs and block parameters
  traversePostOrder(graph->block(), [](Node *node) {
    if (node->kind() != prim::ParallelMap) return true;
    auto block = node->blocks().front();
    for (auto i = 1; i < node->inputs().size();) {
      auto mapIn = node->input(i), blockParam = block->inputs()[i];
      if (!blockParam->hasUses()) {
        node->removeInput(i);
        block->eraseInput(i);
      } else
        i++;
    }
    return true;
  });
  removeDeadRefinedTypes(refinedTypes, graph.get());
}

static void convertMapToLoop(Node *parMap, ValueTypeMap &refinedTypes) {
  // Create loop node
  auto graph = parMap->owningGraph();
  graph->setInsertPoint(parMap->next());
  auto trueCnst = graph->insertConstant(true);
  auto loopNode = graph->create(prim::Loop, {parMap->input(0), trueCnst}, 0);
  loopNode->insertAfter(trueCnst->node());

  // Create list construct nodes
  std::vector<Value *> lists;
  for (auto mapOut : parMap->outputs()) {
    auto elemTy = getElementType(mapOut->type(), 0);
    auto listVal =
        graph->createList(elemTy, {})->insertBefore(loopNode)->output(0);
    lists.push_back(listVal);
    transferRefinedType(mapOut, listVal, refinedTypes);
    mapOut->replaceAllUsesWith(listVal);
  }

  // Create element access nodes
  std::unordered_map<Value *, Value *> valueMap;
  auto mapBlock = parMap->blocks().front(), loopBlock = loopNode->addBlock();
  auto loopIdx = loopBlock->addInput()->setType(IntType::get());
  valueMap.insert({mapBlock->inputs().front(), loopIdx});
  for (auto i : c10::irange(1, mapBlock->inputs().size())) {
    auto mapParam = mapBlock->inputs()[i], inList = parMap->input(i);
    graph->setInsertPoint(loopBlock);
    auto item = graph->insert(aten::__getitem__, {inList, loopIdx})
                    ->setType(mapParam->type());
    valueMap.insert({mapParam, item});
  }

  // Clone nodes to loop body
  cloneNodesToBlock(mapBlock->nodes().front(), mapBlock->nodes().back(),
                    loopBlock, valueMap, &refinedTypes);

  // Add append nodes
  graph->setInsertPoint(loopBlock);
  for (auto i : c10::irange(parMap->outputs().size())) {
    auto ret = mapBlock->outputs()[i], listVal = lists[i];
    graph->insert(aten::append, {listVal, valueMap.at(ret)});
  }
  loopBlock->insertOutput(0, trueCnst);

  // Remove parallel map
  parMap->destroy();
  removeDeadRefinedTypes(refinedTypes, graph);
}

void ConvertInfusibleMapsToLoops(const std::shared_ptr<Graph> &graph,
                                 ValueTypeMap &refinedTypes) {
  // Find infusible parallel maps
  std::vector<Node *> infusibleMaps;
  traversePostOrder(graph->block(), [&](Node *node) {
    if (node->kind() != prim::ParallelMap) return true;
    auto block = node->blocks().front();
    auto frontNode = block->nodes().front();
    if (frontNode->kind() == prim::FusionGroup) return true;
    infusibleMaps.push_back(node);
    return true;
  });

  // Convert each infusible parallel map
  for (auto parMap : infusibleMaps) convertMapToLoop(parMap, refinedTypes);

  // Followup passes
  EliminateCommonSubexprTSSA(graph);
}

static size_t countNodesIn(Block *block) {
  size_t numNodes = 0;
  for (auto node = block->nodes().front(); node != block->nodes().back();
       node = node->next())
    numNodes++;
  return numNodes;
}

static bool isSimpleMap(Node *parMap) {
  // Skip if task number is not constant
  auto body = parMap->blocks().front();
  if (parMap->input(0)->node()->kind() != prim::Constant) return false;

  // Check each node
  for (auto node = body->nodes().front(); node != body->nodes().back();
       node = node->next()) {
    // Skip if there is any nested block
    if (!node->blocks().empty()) return false;

    // Skip if any value defined outside the body is used
    for (auto input : node->inputs()) {
      auto defNode = input->node();
      if (defNode->kind() != prim::Constant && defNode->owningBlock() != body)
        return false;
    }
  }

  return true;
}

static void unrollMap(Node *parMap, ValueTypeMap &refinedTypes) {
  // Prepare for unrolling
  auto graph = parMap->owningGraph();
  graph->setInsertPoint(parMap);
  auto numTasks = *constant_as<int64_t>(parMap->input(0));
  auto body = parMap->blocks().front();

  // Inline body
  std::vector<std::vector<Value *>> allOutputs(
      parMap->outputs().size());  // [output, taskIdx]
  for (auto taskIdx : c10::irange(numTasks)) {
    // Replace parameters with concrete values
    std::unordered_map<Value *, Value *> valueMap;
    auto idxParam = body->inputs()[0];
    auto idxVal = graph->insertConstant(taskIdx);
    valueMap.insert({idxParam, idxVal});
    for (auto argIdx : c10::irange(1, parMap->inputs().size())) {
      auto listArg = parMap->inputs()[argIdx], param = body->inputs()[argIdx];
      auto elem = graph->insert(aten::__getitem__, {listArg, idxVal})
                      ->setType(param->type());
      valueMap.insert({param, elem});
    }

    // Clone body
    cloneNodesTo(body->nodes().front(), body->nodes().back(), parMap, valueMap,
                 &refinedTypes);

    // Collect body returns
    for (auto outIdx : c10::irange(body->outputs().size())) {
      auto ret = body->outputs()[outIdx], output = valueMap.at(ret);
      allOutputs[outIdx].push_back(output);
    }
  }

  // Create output list
  for (auto outIdx : c10::irange(parMap->outputs().size())) {
    auto outList = parMap->output(outIdx);
    auto elemTy = outList->type()->cast<ListType>()->getElementType();
    auto newOutList = graph->createList(elemTy, allOutputs[outIdx])
                          ->insertBefore(parMap)
                          ->output(0);
    outList->replaceAllUsesWith(newOutList);
    transferRefinedType(outList, newOutList, refinedTypes);
  }

  // Remove parallel map
  parMap->destroy();
  removeDeadRefinedTypes(refinedTypes, graph);
}

void UnrollSimpleMaps(const std::shared_ptr<Graph> &graph,
                      ValueTypeMap &refinedTypes) {
  // Collect simple maps
  std::vector<Node *> simpleMaps;
  traversePostOrder(graph->block(), [&](Node *node) {
    if (node->kind() == prim::ParallelMap && isSimpleMap(node))
      simpleMaps.push_back(node);
    return true;
  });

  // Unroll simple maps
  for (auto parMap : simpleMaps) unrollMap(parMap, refinedTypes);
}

void CanonicalizeFusableMaps(const std::shared_ptr<Graph> &graph) {
  traversePreOrder(graph->block(), [](Node *node) {
    // Check if the parallel map is fusable
    if (node->kind() != prim::ParallelMap) return true;
    auto mapBlock = node->blocks().front();
    auto group = mapBlock->nodes().front();
    if (group->kind() != prim::FusionGroup) return true;

    // Check map index
    std::vector<size_t> groupArgPerm;
    auto index = mapBlock->inputs().front();
    auto groupBlock = group->blocks().front();
    if (index->hasUses()) {
      TORCH_CHECK(index->uses().size() == 1);
      groupArgPerm.push_back(index->uses().front().offset);
    } else {
      group->insertInput(0, index)->setType(IntType::get());
      groupBlock->insertInput(0)->setType(IntType::get());
      groupArgPerm.push_back(0);
    }

    // Reorder loop inputs
    for (auto param : mapBlock->inputs().slice(1)) {
      TORCH_CHECK(param->uses().size() == 1);
      groupArgPerm.push_back(param->uses().front().offset);
    }
    for (auto i : c10::irange(group->inputs().size())) {
      if (!std::count(groupArgPerm.begin(), groupArgPerm.end(), i))
        groupArgPerm.push_back(i);
    }
    group->permuteInputs(groupArgPerm);
    groupBlock->permuteInputs(groupArgPerm);

    // Reorder loop outputs
    std::vector<size_t> mapRetPerm;
    for (auto output : group->outputs()) {
      TORCH_CHECK(output->uses().size() == 1);
      mapRetPerm.push_back(output->uses().front().offset);
    }
    mapBlock->permuteOutputs(mapRetPerm);
    node->permuteOutputs(mapRetPerm);

    return true;
  });
}

}  // namespace jit
}  // namespace torch
