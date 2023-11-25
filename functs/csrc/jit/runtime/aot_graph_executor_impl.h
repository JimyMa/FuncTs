#pragma once
#include <c10/util/Flags.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch {
namespace jit {

struct TORCH_API AotGraphExecutorImpl : public GraphExecutorImplBase {
  static std::shared_ptr<Graph> prepareGraph(const std::shared_ptr<Graph>& graph) {
    auto copy = graph->copy();
    graph->dump();
    std::cout << "??" << std::endl;
    // EraseShapeInformation(copy);
    return copy;
  }

  AotGraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name);
  
  void run(Stack& stack) {
    TORCH_CHECK(
      stack.size() >= num_inputs,
      "expected ",
      num_inputs,
      " inputs, but got only ",
      stack.size());

    C10_LOG_API_USAGE_ONCE("torch.graph_executor.run");
    logging::getLogger()->addStatValue(
        logging::runtime_counters::GRAPH_EXECUTOR_INVOCATIONS, 1.0);
    const ExecutionPlan& plan = getPlanFor(stack);
    InterpreterState(plan.code).run(stack);
  }

  c10::intrusive_ptr<Future> runAsync(
    Stack& stack,
    TaskLauncher taskLauncher) {
    TORCH_CHECK(
        stack.size() >= num_inputs,
        "expected ",
        num_inputs,
        " inputs, but got only ",
        stack.size());

    C10_LOG_API_USAGE_ONCE("torch.graph_executor.runAsync");
    logging::getLogger()->addStatValue(
        logging::runtime_counters::GRAPH_EXECUTOR_INVOCATIONS, 1.0);

    struct Frame {
      explicit Frame(ExecutionPlan eplan, TaskLauncher taskLauncher)
          : plan(std::move(eplan)), state(plan.code, std::move(taskLauncher)) {}
      ExecutionPlan plan;
      InterpreterState state;
    };
    auto frame =
        std::make_shared<Frame>(getPlanFor(stack), std::move(taskLauncher));
    auto res = frame->state.runAsync(stack);
    if (!res->completed()) {
      // If not completed, persist the Frame until complete.
      res->addCallback([frame](Future& /* unused */) {});
    }
    return res;
  }

  const ExecutionPlan& getPlanFor(
      Stack& stack,
      c10::optional<size_t> remaining_bailout_depth=0) override;
  GraphExecutorState getDebugState() override;
  ~AotGraphExecutorImpl() override = default;

 private:
  c10::optional<ExecutionPlan> execution_plan_;
  std::shared_ptr<Graph> graph_with_shape;
};

} // namespace jit
} // namespace torch
