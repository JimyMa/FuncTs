//
// Created by jimy on 3/13/22.
//

#ifndef LONG_TAIL_LOGGING_H
#define LONG_TAIL_LOGGING_H

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

inline std::string get_env_variable(char const *env_var_name) {
  if (!env_var_name) {
    return "";
  }
  char *lvl = getenv(env_var_name);
  if (lvl) return std::string(lvl);
  return "";
}

inline int get_log_level() {
  std::string lvl = get_env_variable("LONG_TAIL_LOG_LEVEL");
  return !lvl.empty() ? atoi(lvl.c_str()) : 0;
}

#define LONG_TAIL_ASSERT(Expr, Msg)                                          \
  {                                                                          \
    if (!(Expr)) {                                                           \
      std::cerr << "\033[1;91m"                                              \
                << "[Assertion Failed]"                                      \
                << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line" \
                << __LINE__ << ", Expected :" << #Expr << std::endl;         \
      abort();                                                               \
    }                                                                        \
  }

#define LONG_TAIL_ASSERT_EQ(A, B, Msg, ...) LONG_TAIL_ASSERT((A) == (B), Msg)
#define LONG_TAIL_ASSERT_NE(A, B, Msg, ...) LONG_TAIL_ASSERT((A) == (B), Msg)

#define LONG_TAIL_WARN(Msg)                                                  \
  {                                                                          \
    if (get_log_level() >= 2) {                                              \
      std::cerr << ": \033[1;91m"                                            \
                << "[Warning]"                                               \
                << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line" \
                << __LINE__ << ": " << Msg << std::endl;                     \
    }                                                                        \
  }

#define LONG_TAIL_ABORT(Msg)                                               \
  {                                                                        \
    std::cerr << ": \033[1;91m"                                            \
              << "[Fatal]"                                                 \
              << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line" \
              << __LINE__ << ": " << Msg << std::endl;                     \
    abort();                                                               \
  }

#define LONG_TAIL_LOG_INFO(Msg)                                              \
  {                                                                          \
    if (get_log_level() >= 1) {                                              \
      std::cerr << ": \033[1;91m"                                            \
                << "[Info]"                                                  \
                << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line" \
                << __LINE__ << ": " << Msg << std::endl;                     \
    }                                                                        \
  }

#endif  // LONG_TAIL_LOGGING_H
