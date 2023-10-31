#pragma once

#include "torch/csrc/jit/ir/ir.h"

namespace c10 {
namespace tssa {

static auto SelectSet = Symbol::fromQualString("tssa::SelectSet");
static auto SliceSet = Symbol::fromQualString("tssa::SliceSet");

}  // namespace tssa
}  // namespace c10

namespace torch {
namespace jit {

RegisterOperators registerTssaSetOps();

}
}  // namespace torch
