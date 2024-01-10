#pragma once

#include <string>

#include "functs/csrc/parallel/runtime/cpu/cpu_func_wrapper.h"
#include "functs/csrc/parallel/runtime/module.h"

class CpuModule final : public Module {
  CpuModule(
      std::string data,
      std::string fmt,
      std::string source = "",
      std::string assembly = "")
      : data(data), fmt(fmt), source(assembly), assembly(assembly) {}

  CpuFuncWrapper GetFunction(std::string func_name);

 private:
  std::string data;
  std::string fmt;
  std::string source;
  std::string assembly;
};

