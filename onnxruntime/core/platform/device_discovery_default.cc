// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"
#if defined(__ANDROID__)
#include <dlfcn.h>
#include <vulkan/vulkan.h>
#include <vector>
#endif

namespace onnxruntime {

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  // This is a default implementation.
  // We assume that there is a CPU device and do not attempt to discover anything else.

  std::unordered_set<OrtHardwareDevice> devices{};

  devices.emplace(GetCpuDeviceFromCPUIDInfo());

#if defined(__ANDROID__)
  // M0 compatibility patch: use the Android Vulkan loader's real physical devices.
  // No virtual/software device is advertised, and discovery proves no model placement.
  void* loader = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
  if (!loader) return devices;
  auto create = reinterpret_cast<PFN_vkCreateInstance>(dlsym(loader, "vkCreateInstance"));
  auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(dlsym(loader, "vkDestroyInstance"));
  auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(dlsym(loader, "vkEnumeratePhysicalDevices"));
  auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(dlsym(loader, "vkGetPhysicalDeviceProperties"));
  VkInstance instance = VK_NULL_HANDLE;
  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "M0 ORT discovery";
  app.apiVersion = VK_API_VERSION_1_0;
  VkInstanceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  info.pApplicationInfo = &app;
  if (create && destroy && enumerate && properties && create(&info, nullptr, &instance) == VK_SUCCESS) {
    uint32_t count = 0;
    if (enumerate(instance, &count, nullptr) == VK_SUCCESS && count <= 64) {
      std::vector<VkPhysicalDevice> physical(count);
      if (enumerate(instance, &count, physical.data()) == VK_SUCCESS) {
        for (uint32_t index = 0; index < count; ++index) {
          VkPhysicalDeviceProperties p{};
          properties(physical[index], &p);
          if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU &&
              p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
          OrtHardwareDevice gpu{};
          gpu.type = OrtHardwareDeviceType_GPU;
          gpu.vendor_id = p.vendorID;
          gpu.device_id = p.deviceID;
          gpu.vendor = p.deviceName;
          gpu.metadata.Add("vulkan.driver_version", std::to_string(p.driverVersion));
          gpu.metadata.Add("vulkan.api_version", std::to_string(p.apiVersion));
          devices.emplace(std::move(gpu));
        }
      }
    }
    destroy(instance, nullptr);
  }
  dlclose(loader);
#endif

  return devices;
}

}  // namespace onnxruntime
