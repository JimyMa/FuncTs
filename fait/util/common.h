#pragma once

#include <c10/util/Backtrace.h>
#include <c10/util/Optional.h>

#include <sstream>

namespace torch {
namespace jit {

/* Print to streams */

template <class Stream>
inline void print(Stream &stream) {}

template <class Stream, class Arg, class... Args>
inline void print(Stream &stream, Arg &&arg, Args &&...args) {
  stream << arg;
  print(stream, std::forward<Args>(args)...);
}

template <class Error, class... Args>
inline Error error(Args &&...args) {
  std::stringstream ss;
  print(ss, std::forward<Args>(args)...);
  return Error(ss.str(), c10::get_backtrace(1));
}

/* Optional */

inline bool anyIsNone() { return false; }

template <class T, class... Opts>
inline bool anyIsNone(const c10::optional<T> &opt, const Opts &...opts) {
  if (!opt) return true;
  return anyIsNone(opts...);
}

template <class T, class F, class... Opts>
inline c10::optional<T> tryApply(const F &func, const Opts &...opts) {
  if (anyIsNone(opts...)) return c10::nullopt;
  return func(*opts...);
}

template <class Out, class In, class F>
inline c10::optional<Out> mapOpt(const c10::optional<In> &from, F &&mapFunc) {
  if (from)
    return mapFunc(*from);
  else
    return c10::nullopt;
}

template <class T>
inline c10::optional<T> joinOpt(const c10::optional<T> &accum,
                                const c10::optional<T> &newVal) {
  if (accum)
    return accum;
  else
    return newVal;
}

}  // namespace jit
}  // namespace torch
