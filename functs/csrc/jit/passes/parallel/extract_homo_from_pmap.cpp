#include <cstddef>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include "functs/csrc/jit/utils/ir.h"

#include "passes/parallelize_loops.h"

#include "functs/csrc/parallel/ops/homo_conv.h"

namespace torch {
namespace jit {

int find_block_inputs(Block* b, Value* v) {
  for (size_t i = 0; i < b->inputs().size(); i++) {
    if (b->inputs()[i] == v)
      return i;
  }
  return -1;
}

int find_block_outputs(Block* b, Value* v) {
  for (size_t i = 0; i < b->outputs().size(); i++) {
    if (b->outputs()[i] == v)
      return i;
  }
  return -1;
}

int find_inputs(Node* n, Value* v) {
  for (size_t i = 0; i < n->inputs().size(); i++) {
    if (n->input(i) == v)
      return i;
  }
  return -1;
}

int find_outputs(Node* n, Value* v) {
  for (size_t i = 0; i < n->outputs().size(); i++) {
    if (n->input(i) == v)
      return i;
  }
  return -1;
}

void splitAt(Node* pMap, Node* conv) {
  // convert pMap to pMap + HomoConv2d + pMap
  auto g = pMap->owningGraph();
  auto pMapBlock = pMap->blocks()[0];
  // get parallelMap size
  auto parallel_level = pMap->input(0);
  // step 1: construct parallel_front
  auto parallelFront = g->create(prim::ParallelMap, {parallel_level}, 0);
  auto parallelFrontBlock = parallelFront->addBlock();
  auto frontBlockIter = parallelFrontBlock->addInput();
  frontBlockIter->copyMetadata(pMapBlock->inputs()[0]);

  auto parallelBack = g->create(prim::ParallelMap, {parallel_level}, 0);
  auto parallelBackBlock = parallelBack->addBlock();
  auto backBlockIter = parallelBackBlock->addInput();
  backBlockIter->copyMetadata(pMapBlock->inputs()[0]);

  auto parallelConv = g->create(prim::ParallelMap, {parallel_level}, 0);
  auto parallelConvBlock = parallelConv->addBlock();
  auto convBlockIter = parallelConvBlock->addInput();
  convBlockIter->copyMetadata(pMapBlock->inputs()[0]);

  for (size_t i = 1; i < pMap->inputs().size(); i++) {
    auto addPMapBlockInput = [&](Node* node, Block* block) {
      auto in_ = pMap->input(i);
      node->addInput(in_);
      auto blockInput = block->addInput();
      blockInput->copyMetadata(in_);
    };
    addPMapBlockInput(parallelFront, parallelFrontBlock);
    addPMapBlockInput(parallelBack, parallelBackBlock);
  }

  auto mergeNodeToPMap = [&](Node* to_merge,
                             Node* fusion,
                             std::unordered_map<Value*, Value*>&
                                 pMapBlockValueVsInjectingPMapValue) {
    auto fusionBlock = fusion->blocks()[0];
    auto merged_node = g->create(
        to_merge->kind(), to_merge->inputs(), to_merge->outputs().size());

    merged_node->insertAfter(fusionBlock->param_node());

    for (size_t i = 0; i < to_merge->inputs().size(); i++) {
      auto to_merge_in = to_merge->input(i);
      auto index = find_block_inputs(pMapBlock, to_merge_in);
      // first index is iterator
      if (index > 0) {
        auto fusionInput = fusion->addInput(pMap->input(index));
        auto fusionBlockInput = fusionBlock->addInput();
        fusionBlockInput->copyMetadata(pMapBlock->inputs()[index]);
        pMapBlockValueVsInjectingPMapValue[to_merge_in] = fusionBlockInput;
        merged_node->replaceInput(i, fusionBlockInput);
      } else if (
          to_merge_in->node()->owningBlock() == pMapBlock &&
          !pMapBlockValueVsInjectingPMapValue.count(to_merge_in)) {
        auto fusionInput = fusion->addInput(to_merge_in);
        auto fusionBlockInput = fusionBlock->addInput();
        fusionBlockInput->copyMetadata(to_merge_in);
        pMapBlockValueVsInjectingPMapValue[to_merge_in] = fusionBlockInput;
        merged_node->replaceInput(i, fusionBlockInput);
      } else if (pMapBlockValueVsInjectingPMapValue.count(to_merge_in)) {
        auto fusionBlockInput = pMapBlockValueVsInjectingPMapValue[to_merge_in];
        merged_node->replaceInput(i, fusionBlockInput);
      }
    }

    for (size_t j = 0; j < to_merge->outputs().size(); j++) {
      auto to_merge_out = to_merge->output(j);
      auto merged_node_out = merged_node->output(j);
      if (pMapBlockValueVsInjectingPMapValue.count(to_merge_out)) {
        auto fusionBlockInputSub =
            pMapBlockValueVsInjectingPMapValue[to_merge_out];
        fusionBlockInputSub->replaceAllUsesWith(merged_node_out);
        auto index = find_block_inputs(fusionBlock, fusionBlockInputSub);
        if (index >= 0) {
          fusionBlock->eraseInput(index);
          fusion->removeInput(index);
        }
      }
    }

    for (size_t i = 0; i < to_merge->outputs().size(); i++) {
      merged_node->output(i)->copyMetadata(to_merge->output(i));
      pMapBlockValueVsInjectingPMapValue[to_merge->output(i)] =
          merged_node->output(i);
    }

    for (auto& to_merge_out : to_merge->outputs()) {
      auto index = find_block_outputs(pMapBlock, to_merge_out);
      if (index >= 0) {
        fusionBlock->registerOutput(
            pMapBlockValueVsInjectingPMapValue[to_merge_out]);
        auto fusionOut = fusion->addOutput();
        fusionOut->copyMetadata(pMap->output(index));
        pMap->output(index)->replaceAllUsesAfterNodeWith(pMap, fusionOut);
      }
    }
  };

  bool back = true;
  std::unordered_map<Value*, Value*> pMapBlockValueVsBackPMapValue;
  std::unordered_map<Value*, Value*> pMapBlockValueVsFrontPMapValue;
  std::unordered_map<Value*, Value*> pMapBlockValueVsConvPMapValue;
  for (auto it = pMapBlock->nodes().rbegin(); it != pMapBlock->nodes().rend();
       it++) {
    auto node = *it;
    if (node == conv && back) {
      // construct a homo_conv operator
      mergeNodeToPMap(node, parallelConv, pMapBlockValueVsConvPMapValue);
      back = false;
      continue;
    }
    auto parallelMapInjecting = back ? parallelBack : parallelFront;
    auto ValueMapInjecting =
        back ? pMapBlockValueVsBackPMapValue : pMapBlockValueVsFrontPMapValue;
    mergeNodeToPMap(node, parallelMapInjecting, ValueMapInjecting);
  }
  if (parallelBackBlock->nodes().begin()->next() !=
      parallelBackBlock->param_node())
    parallelBack->insertAfter(pMap);
  parallelConv->insertAfter(pMap);
  if (parallelFrontBlock->nodes().begin()->next() !=
      parallelFrontBlock->param_node())
    parallelFront->insertAfter(pMap);

  auto conv_node = parallelConvBlock->nodes().front();
  auto homo_conv_feat_input = parallelConv->input(1);

  auto homo_conv_node = g->create(
      c10::functs_parallel::HomoConv,
      {parallel_level, homo_conv_feat_input},
      1);
  homo_conv_node->output()->copyMetadata(parallelConv->output());

  for (size_t i = 1; i < conv_node->inputs().size(); i++) {
    homo_conv_node->addInput(conv_node->input(i));
  }

  homo_conv_node->insertAfter(parallelConv);
  parallelConv->output()->replaceAllUsesAfterNodeWith(
      parallelConv, homo_conv_node->output());

  parallelConv->destroy();
  // pMap->destroy();
} // namespace jit

void extractHomoConv(Node* pMap) {
  auto b = pMap->blocks()[0];
  auto ns = b->nodes();
  for (auto n : ns) {
    splitAt(pMap, n);
  }
}

void extractHomoFromPmap(std::shared_ptr<Graph> graph) {
  std::vector<Node*> parMaps;
  traversePostOrder(graph->block(), [&](Node* node) {
    if (node->kind() == prim::ParallelMap)
      parMaps.push_back(node);
    return true;
  });
  for (auto& pMap : parMaps) {
    extractHomoConv(pMap);
    pMap->destroy();
  }
}

} // namespace jit
} // namespace torch
