#include <functs/csrc/jit/runtime/aot_graph_executor.h>

namespace torch {
namespace jit {

AotGraphExecutor::AotGraphExecutor(
    const std::shared_ptr<Graph>& graph,
    std::string function_name)
    : pImpl(std::make_shared<AotGraphExecutorImpl>(graph, std::move(function_name))) {}

void AotGraphExecutor::run(Stack& inputs) {
  return pImpl->run(inputs);
}

c10::intrusive_ptr<Future> AotGraphExecutor::runAsync(
    Stack& stack,
    TaskLauncher taskLauncher) {
  return pImpl->runAsync(stack, std::move(taskLauncher));
}

const ExecutionPlan& AotGraphExecutor::getPlanFor(
    Stack& inputs,
    c10::optional<size_t> remaining_bailout_depth) {
  return pImpl->getPlanFor(inputs, remaining_bailout_depth);
}

GraphExecutorState AotGraphExecutor::getDebugState() {
  return pImpl->getDebugState();
}

// bool AotGraphExecutor::isOptimized() const {
//   return pImpl && pImpl->isOptimized();
// }

// void AotGraphExecutor::debugFlushCompilationCache() {
//   TORCH_INTERNAL_ASSERT(false, "Not Implemented for Legacy Executor");
// }


} // namespace jit
} // namespace torch
