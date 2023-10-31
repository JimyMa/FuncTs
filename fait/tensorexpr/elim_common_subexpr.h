#pragma once

#include "ir_utils.h"

namespace torch {
namespace jit {
namespace tensorexpr {

StmtPtr eliminateCommonSubexpr(StmtPtr stmt);

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch