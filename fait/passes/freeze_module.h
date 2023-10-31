#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

void Freeze(Module* module);

}
}  // namespace torch
