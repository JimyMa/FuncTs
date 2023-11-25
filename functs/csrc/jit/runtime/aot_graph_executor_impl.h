#pragma once
#include <c10/util/Flags.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch {
namespace jit {

struct TORCH_API AotGraphExecutorImpl : public GraphExecutorImplBase {
  static std::shared_ptr<Graph> prepareGraph(const std::shared_ptr<Graph>& graph) {
    auto copy = graph->copy();
    // EraseShapeInformation(copy);
    return copy;
  }

  AotGraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name);

  const ExecutionPlan& getPlanFor(
      Stack& stack,
      c10::optional<size_t> remaining_bailout_depth) override;
  GraphExecutorState getDebugState() override;
  ~AotGraphExecutorImpl() override = default;

 private:
  c10::optional<ExecutionPlan> execution_plan_;
};

} // namespace jit
} // namespace torch
