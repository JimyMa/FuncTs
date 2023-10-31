#include "profile_ops.h"

#include "util/ir.h"

namespace torch {
namespace jit {

static auto _registry = RegisterOperators()
                            .op("prof::Begin(str label) -> ()", profBegin,
                                RegisterOperators::options().aliasAnalysis(
                                    c10::AliasAnalysisKind::CONSERVATIVE))
                            .op("prof::End(str label) -> ()", profEnd,
                                RegisterOperators::options().aliasAnalysis(
                                    c10::AliasAnalysisKind::CONSERVATIVE));

void ConvertProfilingInstrumentation(const std::shared_ptr<Graph> &graph) {
  rewrite(graph->block(), [&](Node *node) -> Node * {
    if (node->kind() != prim::Print) return nullptr;
    if (node->inputs().size() != 2) return nullptr;
    auto label = constant_as<std::string>(node->input(0));
    auto begin = constant_as<bool>(node->input(1));
    if (!label.has_value() || !begin.has_value()) return nullptr;
    auto newNode = *begin ? graph->create(prof::Begin, {node->input(0)}, 0)
                          : graph->create(prof::End, {node->input(0)}, 0);
    TORCH_CHECK(newNode->maybeOperator());
    return replace(node, newNode);
  });
}

}  // namespace jit
}  // namespace torch