#pragma once

#include <memory>
#include "torch/csrc/jit/ir/ir.h"

namespace torch {
namespace jit {
void extractHomoFromPmap(std::shared_ptr<Graph> g);

}
} // namespace torch
