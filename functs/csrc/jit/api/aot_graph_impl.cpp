#include <c10/util/irange.h>
#include <functs/csrc/jit/api/aot_graph_impl.h>
#include <torch/csrc/jit/passes/inliner.h>

#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/peephole.h>

#ifndef C10_MOBILE
#include <ATen/autocast_mode.h>
#include <torch/csrc/jit/passes/autocast.h>
#endif

namespace torch::jit {
namespace {
c10::FunctionSchema defaultSchemaFor(const AotGraphFunction &function) {
  std::vector<c10::Argument> args;
  std::vector<c10::Argument> returns;
  Graph &g = *function.graph();
  size_t num_inputs = function.num_inputs();
  for (const auto i : c10::irange(num_inputs)) {
    const Value *v = g.inputs().at(i);
    std::string name = v->hasDebugName() ? v->debugNameBase()
                                         : ("argument_" + c10::to_string(i));
    args.emplace_back(std::move(name), unshapedType(g.inputs()[i]->type()));
  }
  for (const auto i : c10::irange(g.outputs().size())) {
    returns.emplace_back("", unshapedType(g.outputs()[i]->type()));
  }
  return {function.name(), "", std::move(args), std::move(returns)};
}

template <typename T, typename F>
T *tryToAotGraphFunctionImpl(F &function) noexcept {
  if (!function.isGraphFunction()) {
    return nullptr;
  }

  return static_cast<T *>(&function);
}

template <typename T, typename F> T &toAotGraphFunctionImpl(F &function) {
  if (auto *g = tryToAotGraphFunctionImpl<T>(function)) {
    return *g;
  }

  TORCH_INTERNAL_ASSERT(
      false, "Failed to downcast a Function to a AotGraphFunction. "
             "This probably indicates that the JIT calling context needs a "
             "special case on tryToAotGraphFunction() instead.");
}

} // namespace

static void placeholderCreator(AotGraphFunction &) {
  throw RecursiveMethodCallError();
}

void AotGraphFunction::run(Stack &stack) { get_executor().run(stack); }

c10::intrusive_ptr<c10::ivalue::Future>
AotGraphFunction::runAsync(Stack &stack, TaskLauncher taskLauncher) {
  return get_executor().runAsync(stack, std::move(taskLauncher));
}

void AotGraphFunction::ensure_defined() {
  if (function_creator_) {
    auto creator = function_creator_;
    function_creator_ = placeholderCreator;
    creator(*this);
    function_creator_ = nullptr;
  }
  check_single_output();
}

const c10::FunctionSchema &AotGraphFunction::getSchema() const {
  if (schema_ == nullptr) {
    schema_ = std::make_unique<c10::FunctionSchema>(defaultSchemaFor(*this));
  }
  return *schema_;
}

AotGraphFunction::SpecializationKey
AotGraphFunction::currentSpecialization() const {
  if (force_no_amp_) {
    return SpecializationKey::AutocastOff;
  }
#ifdef C10_MOBILE
  // disabling autodiff pass for mobile build since autocast APIs don't exist
  return SpecializationKey::AutocastOff;
#else
  bool cpu_enabled = at::autocast::is_cpu_enabled();
  bool gpu_enabled = at::autocast::is_enabled();
  if (cpu_enabled && gpu_enabled) {
    return SpecializationKey::CpuGpuAutocastOn;
  } else if (!cpu_enabled && !gpu_enabled) {
    return SpecializationKey::AutocastOff;
  } else {
    return gpu_enabled ? SpecializationKey::GpuAutocastOn
                       : SpecializationKey::CpuAutocastOn;
  }
#endif
}

void preoptimizeGraph(std::shared_ptr<Graph> &graph, bool disable_autocast) {
  Inline(*graph);

  // Peephole Optimize cleans up many "is None" checks and creates constant prop
  // opportunities
  PeepholeOptimize(graph, true);

  // AliasDb construction can be slow, so run it just on immutable types
  // to clean up constant Ifs & other easy wins
  ConstantPropagationImmutableTypes(graph);

#ifndef C10_MOBILE
  // Inject casts for automatic mixed precision
  //
  // TODO: Ideally, this pass could run earlier, before inlining
  //  or any other optimizations. That setup is preferable because:
  //  1. The AMP pass would be self-contained and function independently
  //     of the any optimizations
  //  2. AMP transformations would benefit from followup passes's cleanup
  //
  if (!disable_autocast) {
    Autocast(graph);
  }
#endif

  ConstantPooling(graph);
}

AotGraphFunction *tryToAotGraphFunction(Function &function) noexcept {
  return tryToAotGraphFunctionImpl<AotGraphFunction>(function);
}

AotGraphFunction &toAotGraphFunction(Function &function) {
  return toAotGraphFunctionImpl<AotGraphFunction>(function);
}

const AotGraphFunction &toAotGraphFunction(const Function &function) {
  return toAotGraphFunctionImpl<const AotGraphFunction>(function);
}

} // namespace torch::jit
