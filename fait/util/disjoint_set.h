#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

template <class T>
struct DSElement;

template <class T>
class DisjointSets {
 public:
  DisjointSets() = default;

  void merge(const T &first, const T &second) {
    auto fstPar = findParent(getIndex(first)),
         sndPar = findParent(getIndex(second));
    nodes[sndPar].parent = fstPar;
  }

  bool contains(const T &value) { return valueToNodeIdx.count(value); }

  std::vector<T> getSetOf(const T &value) {
    auto tgtPar = findParent(getIndex(value));
    std::vector<T> set;
    for (auto i = 0u; i < nodes.size(); i++)
      if (findParent(i) == tgtPar) set.push_back(nodes[i].value);
    return set;
  }

  std::vector<T> getAll() {
    std::vector<T> allValues;
    for (auto &pair : valueToNodeIdx) allValues.push_back(pair.first);
    return allValues;
  }

 private:
  size_t getIndex(const T &value) {
    if (valueToNodeIdx.count(value)) return valueToNodeIdx[value];
    auto index = nodes.size();
    nodes.emplace_back(value, index);
    valueToNodeIdx.insert({value, index});
    return index;
  }

  size_t findParent(size_t index) {
    while (nodes[index].parent != index) {
      nodes[index].parent = nodes[nodes[index].parent].parent;
      index = nodes[index].parent;
    }
    return index;
  }

  std::vector<DSElement<T>> nodes;
  std::unordered_map<T, size_t> valueToNodeIdx;
};

template <class T>
struct DSElement {
  T value;
  size_t parent;

  DSElement(const T &value, size_t parent) : value(value), parent(parent) {}
};

}  // namespace jit
}  // namespace torch
