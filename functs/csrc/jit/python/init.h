#pragma once

#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>

TORCH_API extern PyTypeObject THPDtypeType;

namespace torch {
namespace jit {
void initJITFuncTsBindings(PyObject *module);
}
} // namespace torch