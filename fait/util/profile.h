#pragma once

#include <chrono>
#include <string>

namespace torch {
namespace jit {

extern "C" void enableProfiling();
extern "C" void disableProfiling();

void profBegin(const std::string &label);
void profEnd(const std::string &label);

extern "C" void printProfilingResults(size_t count);

std::string fmtDuration(std::chrono::nanoseconds dur);

}  // namespace jit
}  // namespace torch