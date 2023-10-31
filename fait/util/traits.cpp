#include "traits.h"

#include "passes/profile_ops.h"

namespace torch {
namespace jit {

bool hasSideEffects(Node *node) {
  auto sym = node->kind();
  switch (sym) {
    case prim::PythonOp:
    case prim::IgnoredPythonOp:
    case prim::Print:
    case prim::RaiseException:
    case aten::warn:
    case aten::save:
    case aten::manual_seed:
    case prim::AddStatValue:
    case prim::TimePoint:
    case prim::CallFunction:
    case prim::CallMethod:
    case prim::BailoutTemplate:
    case prim::BailOut:
    case prim::rpc_async:
    case prim::rpc_sync:
    case prim::rpc_remote:
    case aten::wait:
    case cuda::set_stream:
    case cuda::_set_device:
    case cuda::_current_device:
    case cuda::synchronize:
    case prim::Enter:
    case prim::Exit:
      return true;
  }

  if (sym == prof::Begin || sym == prof::End) return true;

  return false;
}

}  // namespace jit
}  // namespace torch