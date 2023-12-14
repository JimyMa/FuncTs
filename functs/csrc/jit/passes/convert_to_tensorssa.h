#pragma once

#include <memory>

#include <torch/csrc/jit/ir/ir.h>
#include <functs/csrc/jit/ir/symbol_ext.h>

using namespace c10;

namespace torch {
namespace jit {
struct TensorSSAMutateInfo {
  std::vector<Value*> mutValues{};
  std::unordered_map<Value*, std::vector<Node*>> mutNodes{};

  void addMutNodes(Node* updateNode) {
    AT_ASSERT(
        tensorssa::Update == updateNode->kind(),
        "don't forget we use update node to annotate mutation");

    auto mutated = updateNode->input(1);
    auto mutatee = updateNode->input(0);

    if (!mutNodes.count(mutated)) {
      mutValues.push_back(mutated);
      mutNodes.insert({mutated, std::vector<Node*>()});
    }
    mutNodes[mutated].push_back(updateNode);
  }
};
void DumbRemoveInterPrecedureMutation(std::shared_ptr<Graph> graph);
void ConvertToTensorSSA(std::shared_ptr<Graph> graph);
void TensorSSARemoveUpdate(std::shared_ptr<Graph> graph);
void TensorSSARewriteMutation(std::shared_ptr<Graph> graph, std::shared_ptr<TensorSSAMutateInfo> info);
void TensorSSAPropagation(std::shared_ptr<Graph> graph, std::shared_ptr<TensorSSAMutateInfo> info);
void TensorSSARename(std::shared_ptr<Graph> graph);
} // namespace jit
} // namespace torch
