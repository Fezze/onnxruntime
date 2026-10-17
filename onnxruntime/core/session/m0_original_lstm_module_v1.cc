// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/m0_original_lstm_module_v1.h"
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_MINIMAL_BUILD) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
#include "core/session/plugin_ep/ep_plugin_provider_interfaces.h"
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>
#include "sha256.h"
// Generated after provider strip; session-only dependency, never provider-linked.
#include "m0_qualified_webgpu_provider_v1.inc"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_providers.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/ort_env.h"
#include "core/session/plugin_ep/ep_library_plugin.h"
#include "core/session/onnxruntime_env_config_keys.h"
namespace onnxruntime {
namespace {
void M0OriginalLstmCoreAnchorV1() {}
struct FileLeaseV1 {
  int fd = -1;
  std::filesystem::path path;
  struct stat stamp{};
  ~FileLeaseV1() {
    if (fd >= 0) close(fd);
  }
  void Verify() const {
    struct stat current{}, named{};
    ORT_ENFORCE(fd >= 0 && fstat(fd, &current) == 0 && stat(path.c_str(), &named) == 0,
                "RW07_LSTM_MODULE_FILE_STAT");
    ORT_ENFORCE(S_ISREG(current.st_mode) && current.st_dev == stamp.st_dev && current.st_ino == stamp.st_ino &&
                    named.st_dev == stamp.st_dev && named.st_ino == stamp.st_ino && current.st_size == stamp.st_size &&
                    current.st_mtim.tv_sec == stamp.st_mtim.tv_sec && current.st_mtim.tv_nsec == stamp.st_mtim.tv_nsec &&
                    current.st_ctim.tv_sec == stamp.st_ctim.tv_sec && current.st_ctim.tv_nsec == stamp.st_ctim.tv_nsec,
                "RW07_LSTM_MODULE_FILE_CHANGED");
    model_package::Sha256 hash;
    std::array<char, 65536> bytes;
    off_t offset = 0;
    while (offset < current.st_size) {
      ssize_t count = pread(fd, bytes.data(), bytes.size(), offset);
      if (count < 0 && errno == EINTR) continue;
      ORT_ENFORCE(count > 0, "RW07_LSTM_MODULE_FILE_READ");
      hash.Update(bytes.data(), static_cast<size_t>(count));
      offset += count;
    }
    ORT_ENFORCE(hash.FinalHex() == kM0QualifiedWebGpuProviderSha256V1, "RW07_LSTM_MODULE_DIGEST");
    struct stat after{};
    ORT_ENFORCE(fstat(fd, &after) == 0 && after.st_size == current.st_size &&
                    after.st_mtim.tv_sec == current.st_mtim.tv_sec && after.st_mtim.tv_nsec == current.st_mtim.tv_nsec &&
                    after.st_ctim.tv_sec == current.st_ctim.tv_sec && after.st_ctim.tv_nsec == current.st_ctim.tv_nsec,
                "RW07_LSTM_MODULE_HASH_RACE");
  }
};
}  // namespace
struct M0OriginalLstmModuleV1::Impl {
  FileLeaseV1 file;
  OrtEnvPtr env{nullptr, OrtEnv::Release};
  std::unique_ptr<EpLibraryPlugin> library;
  OrtSessionOptions options;
  std::vector<std::unique_ptr<OrtEpDevice>> devices;
  std::unique_ptr<PluginExecutionProviderFactory> factory;
  std::unique_ptr<ExecutionProviders> providers = std::make_unique<ExecutionProviders>();
  PluginExecutionProvider* provider = nullptr;
  std::shared_ptr<KernelRegistry> registry;
  M0PluginKernelCreatorBindingV1 binding;
  ~Impl() {
    registry.reset();
    providers.reset();
    provider = nullptr;
    factory.reset();
    devices.clear();
    if (library) {
      auto status = library->Unload();
      if (!status.IsOK()) {
        // Retain the failed-unload object/central lease rather than revive its inode.
        LOGS_DEFAULT(ERROR) << "RW07_LSTM_MODULE_UNLOAD: " << status.ErrorMessage();
        (void)library.release();
      }
    }
  }
};
M0OriginalLstmModuleV1::M0OriginalLstmModuleV1(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
M0OriginalLstmModuleV1::~M0OriginalLstmModuleV1() = default;
std::unique_ptr<M0OriginalLstmModuleV1> M0OriginalLstmModuleV1::Open(const logging::Logger& logger) {
  auto impl = std::make_unique<Impl>();
  impl->env = OrtEnv::TryGetInstance();
  ORT_ENFORCE(impl->env, "RW07_LSTM_MODULE_ENV");
  const auto env_config = impl->env->GetEnvironment().GetConfigEntries();
  ORT_ENFORCE(env_config.Entries().find(kOrtEnvAllowVirtualDevices) == env_config.Entries().end() ||
                  env_config.Entries().at(kOrtEnvAllowVirtualDevices) != "1",
              "RW07_LSTM_MODULE_VIRTUAL");
  const char* software = std::getenv("ORT_WEBGPU_EP_ALLOW_SOFTWARE_ADAPTER");
  ORT_ENFORCE(software == nullptr || std::strcmp(software, "1") != 0, "RW07_LSTM_MODULE_SOFTWARE");
  Dl_info core{};
  ORT_ENFORCE(dladdr(reinterpret_cast<void*>(&M0OriginalLstmCoreAnchorV1), &core) != 0 && core.dli_fname,
              "RW07_LSTM_MODULE_CORE_LOCATION");
  const auto core_path = std::filesystem::canonical(core.dli_fname);
  ORT_ENFORCE(core_path.filename() == "libonnxruntime.so" || core_path.filename() == "onnxruntime_preinit_tests",
              "RW07_LSTM_MODULE_CORE_NAME");
  impl->file.path = std::filesystem::canonical(core_path.parent_path() / "libonnxruntime_providers_webgpu.so");
  ORT_ENFORCE(impl->file.path.parent_path() == core_path.parent_path(), "RW07_LSTM_MODULE_SIBLING");
  impl->file.fd = open(impl->file.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  ORT_ENFORCE(impl->file.fd >= 0 && fstat(impl->file.fd, &impl->file.stamp) == 0 &&
                  impl->file.stamp.st_size > 0,
              "RW07_LSTM_MODULE_FILE_OPEN");
  impl->file.Verify();
  impl->library = std::make_unique<EpLibraryPlugin>("rw07-private-original-lstm449-v1", impl->file.path);
  ORT_THROW_IF_ERROR(impl->library->M0LoadOriginalLstmV1());
  impl->file.Verify();
  ORT_THROW_IF_ERROR(impl->library->M0QueryOriginalLstmV1(impl->binding));
  impl->file.Verify();
  OrtEpFactory* selected = nullptr;
  size_t factory_count = 0;
  for (auto* candidate : impl->library->GetFactories()) {
    ++factory_count;
    ORT_ENFORCE(candidate && candidate->GetName && candidate->GetSupportedDevices && candidate->CreateEp &&
                    candidate->ReleaseEp && std::strcmp(candidate->GetName(candidate), kWebGpuExecutionProvider) == 0,
                "RW07_LSTM_MODULE_FACTORY");
    selected = candidate;
  }
  ORT_ENFORCE(factory_count == 1, "RW07_LSTM_MODULE_FACTORY_COUNT");
  const auto& hardware = impl->env->GetEnvironment().GetSortedOrtHardwareDevices();
  std::array<OrtEpDevice*, 16> raw{};
  // Adopt every returned pointer even if the provider reports an error or bad count.
  struct RawGuard {
    std::array<OrtEpDevice*, 16>& raw;
    ~RawGuard() {
      for (size_t i = 0; i < raw.size(); ++i) {
        bool first = true;
        for (size_t j = 0; j < i; ++j) first = first && raw[i] != raw[j];
        if (first) delete raw[i];
      }
    }
  } guard{raw};
  size_t count = 0;
  ORT_THROW_IF_ERROR(ToStatusAndRelease(selected->GetSupportedDevices(selected, hardware.data(), hardware.size(),
                                                                      raw.data(), raw.size(), &count)));
  ORT_ENFORCE(count == 1 && raw[0] && raw[0]->device && raw[0]->ep_factory == selected,
              "RW07_LSTM_MODULE_DEVICE_COUNT");
  for (size_t i = 1; i < raw.size(); ++i) ORT_ENFORCE(raw[i] == nullptr, "RW07_LSTM_MODULE_DEVICE_EXTRA");
  const auto* physical = raw[0]->device;
  ORT_ENFORCE(physical->type == OrtHardwareDeviceType_GPU &&
                  std::find(hardware.begin(), hardware.end(), physical) != hardware.end() &&
                  physical->metadata.Entries().find("is_virtual") == physical->metadata.Entries().end(),
              "RW07_LSTM_MODULE_PHYSICAL_DEVICE");
  impl->devices.emplace_back(raw[0]);
  raw[0] = nullptr;
  impl->options.value.enable_profiling = false;
  impl->options.value.graph_optimization_level = TransformerLevel::Default;
  impl->options.value.use_per_session_threads = true;
  for (const auto* key : {"session.disable_cpu_ep_fallback", "session.disable_prepacking", "session.disable_aot_function_inlining"})
    ORT_THROW_IF_ERROR(impl->options.value.config_options.AddConfigEntry(key, "1"));
  ORT_THROW_IF_ERROR(impl->options.AddProviderOptionsToConfigOptions(
      {{"validationMode", "full"}, {"enableGraphCapture", "0"}, {"enableInt64", "1"}, {"weightLoadAcceleration", "off"}, {"sessionBufferPoolGenerations", "0"}, {"dawnBackendType", "Vulkan"}},
      kWebGpuExecutionProvider));
  const OrtEpDevice* device = impl->devices.front().get();
  impl->factory = std::make_unique<PluginExecutionProviderFactory>(*selected,
                                                                   gsl::span<const OrtEpDevice* const>(&device, 1));
  std::unique_ptr<PluginExecutionProvider> created;
  ORT_THROW_IF_ERROR(impl->factory->CreatePluginExecutionProvider(impl->options, *logger.ToExternal(), created));
  impl->provider = created.get();
  ORT_THROW_IF_ERROR(impl->providers->Add(kWebGpuExecutionProvider,
                                          std::shared_ptr<IExecutionProvider>(std::move(created))));
  ORT_ENFORCE(impl->provider && impl->provider->Type() == kWebGpuExecutionProvider, "RW07_LSTM_MODULE_EP");
  impl->registry = impl->provider->GetKernelRegistry();
  ORT_ENFORCE(impl->registry && impl->binding.callback && impl->binding.state, "RW07_LSTM_MODULE_REGISTRY");
  impl->file.Verify();
  return std::unique_ptr<M0OriginalLstmModuleV1>(new M0OriginalLstmModuleV1(std::move(impl)));
}
PluginExecutionProvider& M0OriginalLstmModuleV1::Provider() const {
  impl_->file.Verify();
  return *impl_->provider;
}
const ExecutionProviders& M0OriginalLstmModuleV1::Providers() const {
  impl_->file.Verify();
  return *impl_->providers;
}
std::shared_ptr<KernelRegistry> M0OriginalLstmModuleV1::Registry() const {
  impl_->file.Verify();
  return impl_->registry;
}
const M0PluginKernelCreatorBindingV1& M0OriginalLstmModuleV1::Binding() const {
  impl_->file.Verify();
  return impl_->binding;
}
}  // namespace onnxruntime
#else
namespace onnxruntime {
struct M0OriginalLstmModuleV1::Impl {};
M0OriginalLstmModuleV1::M0OriginalLstmModuleV1(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
M0OriginalLstmModuleV1::~M0OriginalLstmModuleV1() = default;
std::unique_ptr<M0OriginalLstmModuleV1> M0OriginalLstmModuleV1::Open(const logging::Logger&) { ORT_THROW("RW07_LSTM_MODULE_UNSUPPORTED"); }
PluginExecutionProvider& M0OriginalLstmModuleV1::Provider() const { ORT_THROW("RW07_LSTM_MODULE_UNSUPPORTED"); }
const ExecutionProviders& M0OriginalLstmModuleV1::Providers() const { ORT_THROW("RW07_LSTM_MODULE_UNSUPPORTED"); }
std::shared_ptr<KernelRegistry> M0OriginalLstmModuleV1::Registry() const { ORT_THROW("RW07_LSTM_MODULE_UNSUPPORTED"); }
const M0PluginKernelCreatorBindingV1& M0OriginalLstmModuleV1::Binding() const { ORT_THROW("RW07_LSTM_MODULE_UNSUPPORTED"); }
}  // namespace onnxruntime
#endif
