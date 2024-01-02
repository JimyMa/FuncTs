#include <iostream>

#include <functs/csrc/jit/ir/alias_analysis.h>
#include <functs/csrc/jit/passes/functs_te_fuser.h>

#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/remove_redundant_profiles.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

static OperatorSet FusableOps{
    // Unary operators
    "aten::exp(Tensor self) -> Tensor",
    "aten::sigmoid(Tensor self) -> Tensor",
    "aten::tanh(Tensor self) -> Tensor",
    "aten::relu(Tensor self) -> Tensor",

    // Binary operators
    "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",

    // immutable::Access and immutable::Assign operators
    "immut::access(Tensor src) -> Tensor",
    "immut::assign(Tensor self, Tensor src, bool? n=None) -> Tensor",
    "immut::select(Tensor src, int dim, int index) -> Tensor",
    "immut::select_rev(Tensor self, Tensor src, int dim, int index) -> Tensor ",
    "immut::slice(Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor",
    "immut::slice_rev(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor",
    "immut::squeeze(Tensor self, int dim) -> Tensor",
    "immut::unsqueeze(Tensor self, int dim) -> Tensor",
    "immut::view(Tensor self, int[] size) -> Tensor",
    "immut::reshape(Tensor self, int[] size) -> Tensor",
    "immut::permute(Tensor self, int[] sizes) -> Tensor",
    "immut::expand(Tensor self, int[] size, *, bool implicit) -> Tensor",
    "immut::repeat(Tensor self, int[] size) -> Tensor",
    "immut::expand_as(Tensor self, Tensor other) -> Tensor",
    "immut::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor",

    // MISC
    "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor "};

class FuncTsTensorExprFuser {
 public:
  FuncTsTensorExprFuser(
      std::shared_ptr<Graph> graph,
      size_t min_group_size,
      bool add_composed_op,
      bool fuse_to_dynamic_shapes)
      : graph_(std::move(graph)),
        min_group_size_(min_group_size),
        add_composed_op_(add_composed_op),
        fuse_to_dynamic_shapes_(fuse_to_dynamic_shapes) {}

  bool canHandle(Node* node) {
    return node->isMemberOf(FusableOps);
  }

  size_t blockSize(Block* block) {
    size_t num = 0;
    for (Node* n : block->nodes()) {
      // Don't count prim::Constants and prim::ListConstructs as these are nodes
      // we only pull in along with another, "main", node. E.g. the
      // ListConstruct nodes would also be pulled into a fusion group if they
      // are inputs of an aten::cat node.
      if (n->kind() == prim::Constant || n->kind() == prim::ListConstruct) {
        continue;
      }
      for (Block* b : n->blocks()) {
        num += blockSize(b);
      }
      num++;
    }
    return num;
  }

  bool hasConv(Block* block) {
    for (Node* n : block->nodes()) {
      if (n->kind() == aten::conv2d) {
        return true;
      }
    }
    return false;
  }

  bool inlineIfTooSmall(Node* n) {
    if (n->kind() != prim::TensorExprGroup) {
      return false;
    }
    auto subgraph = n->g(attr::Subgraph);
    size_t num_nodes = blockSize(subgraph->block());
    // Allow small subgraphs containing conv2d, since we'll only select those
    // in cases where the tensorexpr implementation is faster than the aten
    // implementation.
    if (num_nodes < min_group_size_ && !hasConv(subgraph->block())) {
      GRAPH_UPDATE("Fusion group is too small, unmerging: ", *n);
      SubgraphUtils::unmergeSubgraph(n);
      return true;
    }
    // Cleanup the subgraph from duplicated constants while we're at it.
    ConstantPooling(subgraph);

    if (GRAPH_DEBUG_ENABLED) {
      GRAPH_EXPORT("", subgraph);
    }
    return false;
  }

  void inlineSmallFusionGroups(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* n = *it;
      it++;

      for (Block* b : n->blocks()) {
        inlineSmallFusionGroups(b);
      }
      inlineIfTooSmall(n);
    }
  }

  value_list sortReverseTopological(ArrayRef<Value*> inputs, Block* b) {
    value_list result;
    for (auto i : inputs) {
      if (i->node()->owningBlock() == b) {
        result.push_back(i);
      }
    }
    // Sort in reverse topological order
    std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
      return a->node()->isAfter(b->node());
    });
    return result;
  }

#define REQ(cond) \
  if (!(cond)) {  \
    return false; \
  }

  bool canMerge(Node* consumer, Node* producer) {
    // std::cout << "producer: " << *producer << std::endl;
    // std::cout << *producer->next() << std::endl;
    // std::cout << *consumer->prev() << std::endl;
    // std::cout << *consumer->prev()->prev() << std::endl;
    // std::cout << "consumer: " << *consumer << std::endl;
    // Only fuse within a block
    REQ(consumer->owningBlock() == producer->owningBlock());

    // Symbolic checks
    REQ(canHandle(producer) || producer->kind() == prim::TensorExprGroup);
    TORCH_INTERNAL_ASSERT(
        consumer->kind() == prim::TensorExprGroup || canHandle(consumer));

    // nvrtc has a limit on the number of arguments allowed in a CUDA kernel.
    // The specific limit is a function of constant memory size, amount
    // available to pass arguments, and some implementation dependence. Select a
    // safe limit here.
    constexpr size_t subgraphArgLimit = 128;
    auto const nInputs = consumer->inputs().size() +
        consumer->outputs().size() + producer->inputs().size() +
        producer->outputs().size();
    REQ(nInputs <= subgraphArgLimit);

    // Device checks
    // disable device check for debug
    // if (consumer->kind() != aten::cat && producer->kind() != aten::cat) {
    //   // aten::cat needs a special handling because it takes a Tensor[] as
    //   its
    //   // input We deal with that in the code below.
    //   auto consumer_device = tensorexpr::pickDeviceType(consumer->inputs());
    //   REQ(consumer_device);
    //   auto producer_device = tensorexpr::pickDeviceType(producer->inputs());
    //   REQ(producer_device);
    //   REQ(*consumer_device == *producer_device);
    // }

    // Alias checks
    // std::cout << "producer: " << *producer << std::endl;
    // std::cout << *producer->next() << std::endl;
    // std::cout << *consumer->prev() << std::endl;
    // std::cout << *consumer->prev()->prev() << std::endl;
    // std::cout << "consumer: " << *consumer << std::endl;
    REQ(aliasDb_->moveBeforeTopologicallyValid(producer, consumer));

    // Ops that return aliases can only be folded if this is the only use.
    if (producer->kind() == aten::slice ||
        producer->kind() == aten::unsqueeze ||
        producer->kind() == prim::ConstantChunk) {
      for (auto& use : producer->output(0)->uses()) {
        REQ(use.user == consumer);
      }
    }
    if (!consumer->hasAttribute(attr::Subgraph) &&
        consumer->kind() != prim::TensorExprGroup) {
      // Don't initiate a fusion group from prim::ListConstruct
      REQ(consumer->kind() != prim::ListConstruct);
      REQ(consumer->kind() != aten::slice);
      REQ(consumer->kind() != aten::unsqueeze);
      REQ(consumer->kind() != prim::ConstantChunk);

      // Don't initiate a fusion group just for a constant operand
      REQ(producer->kind() != prim::Constant);
    }

    if (producer->kind() == aten::cat) {
      REQ(producer->input(0)->node()->kind() == prim::ListConstruct);
      REQ(producer->input(0)->uses().size() == 1);
      REQ(producer->input(1)->node()->kind() == prim::Constant);
      auto const& listConstruct = producer->input(0)->node();
      // We're merging listconstruct->cat->consumer. cat is the producer here
      // and we cannot determine its device type - we should use device of the
      // listconstruct instead
      auto listconstruct_device =
          tensorexpr::pickDeviceType(listConstruct->inputs());
      auto consumer_device = tensorexpr::pickDeviceType(consumer->inputs());
      REQ(listconstruct_device);
      REQ(consumer_device);
      REQ(*listconstruct_device == *consumer_device);
      for (auto const& input : listConstruct->inputs()) {
        // REQ(isFusableOnDevice(input->node()));
      }
      REQ((nInputs + listConstruct->inputs().size()) <= subgraphArgLimit);
    } else if (consumer->kind() == aten::cat) {
      REQ(consumer->input(0)->node()->kind() == prim::ListConstruct);
      REQ(consumer->input(0)->uses().size() == 1);
      REQ(consumer->input(1)->node()->kind() == prim::Constant);
      auto const& listConstruct = consumer->input(0)->node();
      // We're merging listconstruct->cat. cat is the consumer and listconstruct
      // is the producer. cat doesn't have its device type and thus the only
      // thing we should check is that listconstruct's device is well defined
      // (e.g. all its inputs has the same device).
      auto listconstruct_device =
          tensorexpr::pickDeviceType(listConstruct->inputs());
      REQ(listconstruct_device);
      REQ((nInputs + listConstruct->inputs().size()) <= subgraphArgLimit);
    } else {
      // REQ(isFusableOnDevice(producer));
    }

    return true;
  }
#undef REQ

  bool canFuseOnDevice(Value* v) {
    auto type = v->type()->cast<TensorType>();
    if (!type) {
      return true;
    }
    auto device = type->device();
    if (!device) {
      return false;
    }
    if (device->is_cpu()) {
      return canFuseOnCPU();
    } else if (device->is_cuda()) {
      // #ifndef C10_MOBILE
      //       if (fuser::cuda::isEnabled()) {
      //         return false;
      //       }
      // #endif
      return canFuseOnGPU();
    } else if (device->is_xpu()) {
      return false;
    }
    return false;
  }

  bool isFusableOnDevice(Node* node) {
    for (const auto& input : node->inputs()) {
      if (input->node()->kind() == prim::ListConstruct) {
        if (!isFusableOnDevice(input->node())) {
          return false;
        }
      }
      if (!canFuseOnDevice(input)) {
        return false;
      }
    }
    return true;
  }

  c10::optional<Node*> tryMerge(Node* fusion_group, Node* to_merge) {
    // std::cout << "producer: " << *to_merge << std::endl;
    // std::cout << *producer->next() << std::endl;
    // std::cout << *consumer->prev() << std::endl;
    // std::cout << *consumer->prev()->prev() << std::endl;
    // std::cout << "consumer: " << *fusion_group << std::endl;
    if (!canMerge(fusion_group, to_merge)) {
      return c10::nullopt;
    }

    std::vector<Node*> nodes_to_merge = {to_merge};

    if (to_merge->kind() == aten::cat) {
      Node* listconstruct = to_merge->input(0)->node();
      nodes_to_merge.push_back(listconstruct);
    }

    // First, try to move all the nodes we want to fuse next to the fusion
    // group.
    Node* move_point = fusion_group;
    for (auto n : nodes_to_merge) {
      GRAPH_UPDATE("Trying to move node next to fusion group: ", getHeader(n));
      if (!aliasDb_->moveBeforeTopologicallyValid(n, move_point)) {
        GRAPH_UPDATE("Failed to move because of AliasDB checks!");
        return c10::nullopt;
      }
      move_point = n;
    }

    // Now all the nodes that we're going to fuse are moved next to the fusion
    // group, so we can safely merge them into the fusion group subgraph.
    fusion_group = getOrCreateTensorExprSubgraph(fusion_group);

    for (auto n : nodes_to_merge) {
      GRAPH_UPDATE("Merging ", getHeader(n));
      SubgraphUtils::mergeNodeIntoSubgraphAndUpdateAliasing(
          n, fusion_group, *aliasDb_);
    }
    return fusion_group;
  }

  // Create a fusion group starting from the node N.
  // We then try to pull inputs into the fusion group and repeat that process
  // until there is nothing we can pull in.
  // Step 1: create a TensorExprNode which contains node
  // Step 1.1: change the producer to TensorExprNode which origin producer is
  // node. Step 2: check if the producer of node can be merged to
  // TensorExprNode Step 2.1: add fusable producer to TensorExprNode Step 2.2:
  // adjust input and output Step 3: return new iterator which points to the
  // node->prev()
  std::pair<graph_node_list::iterator, bool> createFusionGroup(
      Node* fusion_node) {
    // Allow single-node groups containing conv2d, since we'll only select
    // those in cases where the tensorexpr implementation is faster than the
    // aten implementation.
    if (min_group_size_ == 1 || fusion_node->kind() == aten::conv2d) {
      fusion_node = getOrCreateTensorExprSubgraph(fusion_node);
    }

    GRAPH_DEBUG("Iteratively pull input nodes into the fusion group...\n");
    auto inputs = sortReverseTopological(
        fusion_node->inputs(), fusion_node->owningBlock());
    for (auto input : inputs) {
      debugDumpFusionGroup("Current fusion group: ", fusion_node);
      GRAPH_DEBUG("Trying to merge: ", *input->node());
      if (auto maybe_fusion_group = tryMerge(fusion_node, input->node())) {
        // we successfully merged, so the new group's `inputs` may have
        // changed. So rescan the new group for more merging opportunities.
        return std::make_pair(
            maybe_fusion_group.value()->reverseIterator(), true);
      }
    }

    return std::make_pair(++fusion_node->reverseIterator(), false);
  }

  std::pair<graph_node_list::iterator, bool> scanNode(Node* n) {
    if (!canHandle(n)) {
      return std::make_pair(++n->reverseIterator(), false);
    }
    // Begin to perform kernel fusion
    return createFusionGroup(n);
  }

  static void debugDumpFusionGroup(const std::string& msg, Node* n) {
    // NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
    GRAPH_DEBUG(msg, *n);
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    if (n->kind() == prim::TensorExprGroup) {
      GRAPH_DEBUG(*n->g(attr::Subgraph));
    }
  }

  void createFusionGroups(Block* block) {
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        bool changed;
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        createFusionGroups(b);
      }
    }

    // Try to merge adjacent fusion groups together. Because we have only merged
    // by looking at graph inputs, without this we would not attempt to merge
    // adjacent fusion groups that don't have a dependency on each other

    std::vector<Node*> initial_fusion_groups;
    for (Node* n : block->nodes()) {
      if (n->kind() == prim::TensorExprGroup) {
        initial_fusion_groups.push_back(n);
      }
    }

    Node* prev_fusion_group =
        !initial_fusion_groups.empty() ? initial_fusion_groups[0] : nullptr;

    for (const auto i : c10::irange(1, initial_fusion_groups.size())) {
      // Try merging the just created fusion group into the previous one.
      // If it did not work, then put the previous fusion group into
      // fusion_groups vector - we will not touch it anymore in this loop.
      // If merging succeeded, save the merged group as the "previous" fusion
      // group so that we can try to merge the next one into it.

      Node* fusion_group = initial_fusion_groups[i];
      debugDumpFusionGroup(
          "Trying to merge into the previous fusion group:", prev_fusion_group);
      // std::cout << "prev_fusion_group: " << *prev_fusion_group << std::endl;
      // std::cout << "fusion_group:" << *fusion_group << std::endl;
      if (auto merged_fusion_group =
              tryMerge(fusion_group, prev_fusion_group)) {
        // substitute tryMerge(prev_fusion_group, fusion_group) for AliasDb
        // TORCH_API compatibility
        prev_fusion_group = *merged_fusion_group;
        debugDumpFusionGroup(
            "Successfully merged into the previous fusion group: ",
            prev_fusion_group);
      } else {
        GRAPH_DEBUG("Cannot merge into the previous fusion group");
        prev_fusion_group = fusion_group;
      }
    }
  }

  void run() {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
    aliasDbCopy_ = torch::make_unique<AliasDbCopy>(graph_);
    RemoveRedundantProfiles(graph_);
    GRAPH_DUMP("After removing redundant profile nodes: ", graph_);
    createFusionGroups(graph_->block());
    GRAPH_DUMP("After creating fusion groups: ", graph_);
    // we maintain alias db correctness during initial fusion, but it is
    // difficult to maintain correctness after inlining so inline only after
    // fusion is done.
    inlineSmallFusionGroups(graph_->block());
    GRAPH_DUMP("After inlining small fusion groups: ", graph_);
    // if (fuse_to_dynamic_shapes_) {
    //   VLOG(1) << "TensorExpr fusion with dynamic shapes is enabled"
    //           << std::endl;
    //   generalizeFusionGroups(graph_->block());
    //   GRAPH_DUMP("After generalizing fusion groups: ", graph_);
    // } else {
    //   prepareFusionGroupAndGuardOutputs(graph_->block());
    //   GRAPH_DUMP("After guarding fusion groups: ", graph_);
    // }
  }

 private:
  Node* getOrCreateTensorExprSubgraph(Node* n) {
    if (n->hasAttribute(attr::Subgraph) && n->kind() == prim::TensorExprGroup) {
      return n;
    }
    GRAPH_UPDATE("Creating a tensorexpr::Group node from: ", *n);
    return SubgraphUtils::createSingletonSubgraphAndUpdateAliasing(
        n, prim::TensorExprGroup, *aliasDb_);
  }

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::unique_ptr<AliasDbCopy> aliasDbCopy_ = nullptr;

  std::set<NodeKind> operators_not_to_fuse;
  // Minimal size of a fusion group
  size_t min_group_size_;
  // compose Runtime Type Guard and Kernel in one op
  bool add_composed_op_;
  // generalize static shapes to dynamic shapes
  bool fuse_to_dynamic_shapes_;
};

void FuncTsFuseTensorExprs(
    std::shared_ptr<Graph>& graph,
    size_t min_group_size,
    bool add_composed_op,
    bool fuse_to_dynamic_shapes) {
  GRAPH_DUMP("Before TExprFuser: ", graph);

  // Temporary change for Block code generation.
  if (tensorexpr::getTEGenerateBlockCode()) {
    min_group_size = 1;
  }

  if (add_composed_op) {
    TORCH_INTERNAL_ASSERT(
        fuse_to_dynamic_shapes, "Fusing static shapes with composed op NYI");
  }

  // Get rid of dead code so that we don't waste effort fusing it.
  EliminateDeadCode(graph);

  FuncTsTensorExprFuser fuser(
      graph, min_group_size, add_composed_op, fuse_to_dynamic_shapes);
  fuser.run();

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

  GRAPH_DUMP("After TExprFuser: ", graph);
}

// TODO: to graph builder
// RegisterOperators TensorExprOps({
//     torch::jit::Operator(
//         prim::TensorExprGroup,
//         createTensorExprOp,
//         AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
// });

} // namespace jit
} // namespace torch
