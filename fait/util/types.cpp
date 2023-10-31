#include "types.h"

namespace torch {
namespace jit {

#define NAME_TO_DTYPE(_, dtype) {#dtype, c10::k##dtype},

std::unordered_map<std::string, c10::ScalarType> strToDtype{
    AT_FORALL_SCALAR_TYPES(NAME_TO_DTYPE)};

}  // namespace jit
}  // namespace torch