#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "functs/csrc/parallel/runtime/device_api.h"

#include "functs/csrc/utils/logging.h"

namespace runtime {

class Registry {
 public:
  typedef std::shared_ptr<DeviceAPI> (*Creator)();
  typedef std::map<std::string, Creator> RegistryManager;
  static RegistryManager& Manager() {
    static RegistryManager* g_registry_ = new RegistryManager();
    return *g_registry_;
  }
  static void Register(const std::string& name, Creator creator) {
    RegistryManager& m = Manager();
    LONG_TAIL_ASSERT(
        m.count(name) == 0,
        "Device type" << name.c_str() << " already registered.");
    m[name] = creator;
  }
  static std::shared_ptr<DeviceAPI> Get(const std::string& name) {
    RegistryManager& m = Manager();
    auto it = m.find(name);
    if (it == m.end())
      LONG_TAIL_WARN(
          "No Device API Found," << name.c_str()
                                 << " is not Registered, Please check your "
                                    "cmake config, Device Name: ");
    return it->second();
  }

  static std::vector<std::string> DeviceList() {
    std::vector<std::string> result;
    RegistryManager& m = Manager();
    for (auto iter = m.begin(); iter != m.end(); ++iter) {
      result.push_back(iter->first);
    }
    return result;
  }

  static std::string DeviceListString() {
    std::string device_types_str;
    RegistryManager& m = Manager();
    for (auto iter = m.begin(); iter != m.end(); ++iter) {
      if (iter != m.begin()) {
        device_types_str += ", ";
      }
      device_types_str += iter->first;
    }
    return device_types_str;
  }

 private:
  Registry() {}
};

class DeviceRegisterer {
 public:
  DeviceRegisterer(
      const std::string& type,
      std::shared_ptr<DeviceAPI> (*creator)()) {
    // ELENA_LOG_INFO("Registering Device type: %s", type.c_str());
    Registry::Register(type, creator);
  }
};

#define REGISTER_DEVICE_CREATOR(type, creator) \
  static DeviceRegisterer g_creator_##type(#type, creator);

#define REGISTER_DEVICE(type)                               \
  std::shared_ptr<DeviceAPI> Creator_##type##DeviceAPI() {  \
    return std::shared_ptr<DeviceAPI>(new type##DeviceAPI); \
  }                                                         \
  REGISTER_DEVICE_CREATOR(type, Creator_##type##DeviceAPI)
} // namespace runtime
