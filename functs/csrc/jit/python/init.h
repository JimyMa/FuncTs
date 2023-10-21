#pragma once

#include <Python.h>

namespace torch {
namespace jit {
void initJITFuncBindings(PyObject* module);
}
}