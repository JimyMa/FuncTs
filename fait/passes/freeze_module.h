#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

void Freeze(Module *module);
Module Clone(Module *module);

} // namespace jit
} // namespace torch
