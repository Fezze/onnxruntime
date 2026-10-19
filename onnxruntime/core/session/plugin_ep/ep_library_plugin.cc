// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_library_plugin.h"

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/session/environment.h"
#include "core/session/plugin_ep/ep_library_plugin_utils.h"

#if defined(__ANDROID__) && defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <filesystem>
namespace {
struct M0ModuleLeaseV1 {
  const void* exclusive = nullptr;
  std::set<const void*> shared;
};
std::mutex& M0LeaseMutexV1() {
  static auto* mutex = new std::mutex;
  return *mutex;
}
std::map<std::string, M0ModuleLeaseV1>& M0LeasesV1() {
  static auto* leases = new std::map<std::string, M0ModuleLeaseV1>;
  return *leases;
}
[[maybe_unused]] std::string M0FileKeyV1(const std::filesystem::path& path) {
  struct stat st{};
  ORT_ENFORCE(stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode), "RW07_LSTM_LEASE_FILE");
  return std::to_string(st.st_dev) + ":" + std::to_string(st.st_ino);
}
std::string M0MappedKeyV1(void* symbol) {
  const auto address = reinterpret_cast<uintptr_t>(symbol);
  std::ifstream maps("/proc/self/maps");
  ORT_ENFORCE(maps.is_open(), "RW07_LSTM_MODULE_MAPS");
  std::string line;
  while (std::getline(maps, line)) {
    std::istringstream fields(line);
    std::string range, permissions, offset, device, inode;
    if (!(fields >> range >> permissions >> offset >> device >> inode)) continue;
    const auto dash = range.find('-');
    const auto colon = device.find(':');
    ORT_ENFORCE(dash != std::string::npos && colon != std::string::npos, "RW07_LSTM_MODULE_MAP_FORMAT");
    const auto begin = std::stoull(range.substr(0, dash), nullptr, 16);
    const auto end = std::stoull(range.substr(dash + 1), nullptr, 16);
    if (address < begin || address >= end) continue;
    const auto dev = makedev(static_cast<unsigned int>(std::stoul(device.substr(0, colon), nullptr, 16)),
                             static_cast<unsigned int>(std::stoul(device.substr(colon + 1), nullptr, 16)));
    ORT_ENFORCE(std::stoull(inode) > 0, "RW07_LSTM_MODULE_ANONYMOUS");
    return std::to_string(dev) + ":" + inode;
  }
  ORT_THROW("RW07_LSTM_MODULE_UNMAPPED_SYMBOL");
}
onnxruntime::Status M0AcquireLeaseV1(const void* owner, const std::string& key, bool exclusive) {
  std::lock_guard<std::mutex> lock(M0LeaseMutexV1());
  auto& lease = M0LeasesV1()[key];
  if (exclusive) {
    ORT_RETURN_IF(lease.exclusive != nullptr || !lease.shared.empty(), "RW07_LSTM_MODULE_ALREADY_LEASED");
    lease.exclusive = owner;
  } else {
    ORT_RETURN_IF(lease.exclusive != nullptr && lease.exclusive != owner, "RW07_LSTM_MODULE_EXCLUSIVE");
    if (lease.exclusive == nullptr) lease.shared.insert(owner);
  }
  return onnxruntime::Status::OK();
}
void M0ReleaseLeaseV1(const void* owner, const std::string& key) {
  if (key.empty()) return;
  std::lock_guard<std::mutex> lock(M0LeaseMutexV1());
  auto found = M0LeasesV1().find(key);
  if (found == M0LeasesV1().end()) return;
  auto& lease = found->second;
  if (lease.exclusive == owner) lease.exclusive = nullptr;
  lease.shared.erase(owner);
  if (lease.exclusive == nullptr && lease.shared.empty()) M0LeasesV1().erase(found);
}
[[maybe_unused]] std::filesystem::path M0SymbolPathV1(void* symbol, void** base = nullptr) {
  Dl_info info{};
  ORT_ENFORCE(symbol && dladdr(symbol, &info) && info.dli_fname && info.dli_fbase, "RW07_LSTM_MODULE_SYMBOL");
  if (base) *base = info.dli_fbase;
  return std::filesystem::canonical(info.dli_fname);
}
}  // namespace
#endif
namespace onnxruntime {
Status EpLibraryPlugin::Load() {
  auto status = Status::OK();

  ORT_TRY {
    std::lock_guard<std::mutex> lock{mutex_};
    if (factories_.empty()) {
      // Run the load steps in an inner lambda so that an early error return from ORT_RETURN_IF_ERROR
      // is captured in `status` rather than returning out of this function. That lets the cleanup
      // below run on ANY failure path (error status or exception). Previously the ORT_RETURN_IF_ERROR
      // returns bypassed the ORT_CATCH cleanup, so a partial load - e.g. the library handle was
      // acquired but a required export was missing or CreateEpFactories failed - left the library
      // loaded and half-initialized.
      status = [&]() -> Status {
        ORT_RETURN_IF_ERROR(Env::Default().LoadDynamicLibrary(library_path_, false, &handle_));
        ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_, "CreateEpFactories",
                                                                reinterpret_cast<void**>(&create_fn_)));
        ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_, "ReleaseEpFactory",
                                                                reinterpret_cast<void**>(&release_fn_)));

#if defined(__ANDROID__) && defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
        // The lease is held through provider callbacks and unload; the map lock is not.
        const auto lease_key = M0MappedKeyV1(reinterpret_cast<void*>(create_fn_));
        ORT_RETURN_IF_ERROR(M0AcquireLeaseV1(this, lease_key, false));
        m0_module_lease_key_ = lease_key;
#endif
        return ep_library_plugin_utils::CreateFactories(create_fn_, release_fn_, registration_name_, factories_);
      }();
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      // TODO: Add logging of exception
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load execution provider library: ", library_path_,
                               " with error: ", ex.what());
    });
  }

  // If the load failed for any reason (error status or exception), unload the library so a partial
  // load does not leave it loaded and half-initialized.
  if (!status.IsOK()) {
    auto unload_status = Unload();
    if (!unload_status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Failed to unload execution provider library: " << library_path_ << " with error: "
                          << unload_status.ErrorMessage();
    }
  }

  return status;
}

Status EpLibraryPlugin::Unload() {
  std::lock_guard<std::mutex> lock{mutex_};

  if (handle_) {
    factories_.clear();

    ORT_RETURN_IF_ERROR(Env::Default().UnloadDynamicLibrary(handle_));
  }

  handle_ = nullptr;
  m0_query_v1_ = nullptr;
#if defined(__ANDROID__) && defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
  M0ReleaseLeaseV1(this, m0_module_lease_key_);
  m0_module_lease_key_.clear();
#endif

  return Status::OK();
}

Status EpLibraryPlugin::M0LoadOriginalLstmV1() {
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_MINIMAL_BUILD) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
  auto status = Status::OK();
  ORT_TRY {
    std::lock_guard<std::mutex> lock{mutex_};
    status = [&]() -> Status {
      ORT_RETURN_IF(handle_ != nullptr || !factories_.empty() || !m0_module_lease_key_.empty(), "RW07_LSTM_MODULE_REUSE");
      const auto canonical = std::filesystem::canonical(library_path_);
      const auto key = M0FileKeyV1(canonical);
      if (void* foreign = dlopen(canonical.c_str(), RTLD_NOW | RTLD_NOLOAD)) {
        dlclose(foreign);
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_LSTM_MODULE_FOREIGN_LOAD");
      }
      ORT_RETURN_IF_ERROR(M0AcquireLeaseV1(this, key, true));
      m0_module_lease_key_ = key;
      ORT_RETURN_IF_ERROR(Env::Default().LoadDynamicLibrary(canonical, false, &handle_));
      ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_, "CreateEpFactories", reinterpret_cast<void**>(&create_fn_)));
      ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_, "ReleaseEpFactory", reinterpret_cast<void**>(&release_fn_)));
      M0WebGpuLstm14CreatorBindingQueryV1 query = nullptr;
      ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_, "OrtM0WebGpuLstm14CreatorBindingV1", reinterpret_cast<void**>(&query)));
      void* base = nullptr;
      ORT_RETURN_IF_NOT(M0SymbolPathV1(reinterpret_cast<void*>(query), &base) == canonical &&
                            M0FileKeyV1(canonical) == key && M0MappedKeyV1(reinterpret_cast<void*>(query)) == key,
                        "RW07_LSTM_MODULE_QUERY_ORIGIN");
      for (void* symbol : {reinterpret_cast<void*>(create_fn_), reinterpret_cast<void*>(release_fn_)}) {
        void* other = nullptr;
        ORT_RETURN_IF_NOT(M0SymbolPathV1(symbol, &other) == canonical && other == base && M0MappedKeyV1(symbol) == key,
                          "RW07_LSTM_MODULE_FACTORY_ORIGIN");
      }
      m0_query_v1_ = reinterpret_cast<void*>(query);
      return Status::OK();
    }();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() { status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_LSTM_MODULE_OPEN: ", ex.what()); });
  }
  if (!status.IsOK()) {
    const auto cleanup = Unload();
    if (!cleanup.IsOK()) {
      LOGS_DEFAULT(ERROR) << cleanup.ErrorMessage();
    }
  }
  return status;
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_LSTM_MODULE_UNSUPPORTED");
#endif
}

Status EpLibraryPlugin::M0QueryOriginalLstmV1(M0PluginKernelCreatorBindingV1& binding) {
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_MINIMAL_BUILD) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
  std::lock_guard<std::mutex> lock{mutex_};
  ORT_RETURN_IF_NOT(handle_ && m0_query_v1_ && !m0_module_lease_key_.empty() && factories_.empty(),
                    "RW07_LSTM_MODULE_QUERY_STATE");
  const auto canonical = std::filesystem::canonical(library_path_);
  void* base = nullptr;
  ORT_RETURN_IF_NOT(M0SymbolPathV1(m0_query_v1_, &base) == canonical &&
                        M0FileKeyV1(canonical) == m0_module_lease_key_,
                    "RW07_LSTM_MODULE_QUERY_CHANGED");
  ORT_RETURN_IF_ERROR(ep_library_plugin_utils::CreateFactories(create_fn_, release_fn_, registration_name_, factories_));
  const auto query = reinterpret_cast<M0WebGpuLstm14CreatorBindingQueryV1>(m0_query_v1_);
  M0PluginKernelCreatorBindingV1 actual;
  ORT_RETURN_IF_NOT(query(1, &actual) == 1 && actual.callback && actual.state, "RW07_LSTM_MODULE_BINDING");
  for (void* symbol : {reinterpret_cast<void*>(actual.callback), actual.state}) {
    void* other = nullptr;
    ORT_RETURN_IF_NOT(M0SymbolPathV1(symbol, &other) == canonical && other == base && M0MappedKeyV1(symbol) == m0_module_lease_key_,
                      "RW07_LSTM_MODULE_BINDING_ORIGIN");
  }
  ORT_RETURN_IF_NOT(M0FileKeyV1(canonical) == m0_module_lease_key_, "RW07_LSTM_MODULE_INODE_CHANGED");
  binding = actual;
  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(binding);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "RW07_LSTM_MODULE_UNSUPPORTED");
#endif
}
Status EpLibraryPlugin::M0QueryNativeKernelV2(uint32_t record,M0NativeProviderKernelBindingV2& binding) {
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_MINIMAL_BUILD) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
  auto status=Status::OK();
  ORT_TRY {
    std::lock_guard<std::mutex> lock{mutex_};
    status=[&]()->Status{
      ORT_RETURN_IF_NOT(handle_&&m0_query_v1_&&!m0_module_lease_key_.empty()&&factories_.size()==1&&M0NativeKernelRecordByIdV2(record),
                        "RW07_NATIVE_MODULE_QUERY_STATE");
      const auto canonical=std::filesystem::canonical(library_path_);
      void* base=nullptr;
      ORT_RETURN_IF_NOT(M0SymbolPathV1(m0_query_v1_,&base)==canonical&&M0MappedKeyV1(m0_query_v1_)==m0_module_lease_key_&&
                            M0FileKeyV1(canonical)==m0_module_lease_key_,"RW07_NATIVE_MODULE_ORIGIN_CHANGED");
      M0WebGpuNativeCreatorBindingQueryV2 query=nullptr;
      ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_,"OrtM0WebGpuNativeCreatorBindingV2",reinterpret_cast<void**>(&query)));
      void* query_base=nullptr;
      ORT_RETURN_IF_NOT(query&&M0SymbolPathV1(reinterpret_cast<void*>(query),&query_base)==canonical&&query_base==base&&
                            M0MappedKeyV1(reinterpret_cast<void*>(query))==m0_module_lease_key_,"RW07_NATIVE_MODULE_QUERY_ORIGIN");
      M0NativeProviderKernelBindingV2 pending;
      ORT_RETURN_IF_NOT(query(2,record,&pending)==1&&pending.abi==2&&pending.record_id==record&&pending.registry&&
                            pending.registration_definition&&pending.creator.callback&&pending.creator.state,
                        "RW07_NATIVE_MODULE_QUERY_BINDING");
      for(void* symbol:{reinterpret_cast<void*>(pending.creator.callback),pending.creator.state}){
        void* actual_base=nullptr;
        ORT_RETURN_IF_NOT(M0SymbolPathV1(symbol,&actual_base)==canonical&&actual_base==base&&M0MappedKeyV1(symbol)==m0_module_lease_key_,
                          "RW07_NATIVE_MODULE_CREATOR_ORIGIN");
      }
      ORT_RETURN_IF_NOT(M0FileKeyV1(canonical)==m0_module_lease_key_,"RW07_NATIVE_MODULE_QUERY_INODE_CHANGED");
      binding=pending;return Status::OK();
    }();
  }
  ORT_CATCH(const std::exception& error){
    ORT_HANDLE_EXCEPTION([&](){status=ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,"RW07_NATIVE_MODULE_QUERY: ",error.what());});
  }
  return status;
#else
  ORT_UNUSED_PARAMETER(record);ORT_UNUSED_PARAMETER(binding);
  return ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,"RW07_NATIVE_MODULE_UNSUPPORTED");
#endif
}
Status EpLibraryPlugin::M0QueryNativeResourceV3(const OrtEp* ep,uintptr_t handle,uint64_t required_bytes,
                                               M0NativeResourceBindingV3& binding){
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && defined(__ANDROID__) && defined(USE_WEBGPU) && !defined(ORT_MINIMAL_BUILD) && !defined(ORT_PLUGIN_EP_STATICALLY_LINKED)
 auto status=Status::OK();
 ORT_TRY {
  std::lock_guard<std::mutex> lock{mutex_};
  status=[&]()->Status{
   ORT_RETURN_IF_NOT(handle_&&m0_query_v1_&&!m0_module_lease_key_.empty()&&factories_.size()==1&&ep,
     "RW07_NATIVE_RESOURCE_QUERY_STATE");
   const auto canonical=std::filesystem::canonical(library_path_);void* base=nullptr;
   ORT_RETURN_IF_NOT(M0SymbolPathV1(m0_query_v1_,&base)==canonical&&M0MappedKeyV1(m0_query_v1_)==m0_module_lease_key_&&
     M0FileKeyV1(canonical)==m0_module_lease_key_,"RW07_NATIVE_RESOURCE_MODULE_ORIGIN_CHANGED");
   M0WebGpuNativeResourceQueryV3 query=nullptr;
   ORT_RETURN_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_,"OrtM0WebGpuNativeResourceBindingV3",reinterpret_cast<void**>(&query)));
   void* query_base=nullptr;
   ORT_RETURN_IF_NOT(query&&M0SymbolPathV1(reinterpret_cast<void*>(query),&query_base)==canonical&&query_base==base&&
     M0MappedKeyV1(reinterpret_cast<void*>(query))==m0_module_lease_key_,"RW07_NATIVE_RESOURCE_QUERY_ORIGIN");
   // This Ep was created and retained by M0OriginalLstmModuleV1's own authenticated
   // factory. Its concrete dispatch entry points must belong to the same mapping.
   for(void* symbol:{reinterpret_cast<void*>(ep->GetName),reinterpret_cast<void*>(ep->GetKernelRegistry),
                    reinterpret_cast<void*>(ep->OnRunStart),reinterpret_cast<void*>(ep->OnRunEnd)}){
    void* symbol_base=nullptr;
    ORT_RETURN_IF_NOT(symbol&&M0SymbolPathV1(symbol,&symbol_base)==canonical&&symbol_base==base&&
      M0MappedKeyV1(symbol)==m0_module_lease_key_,"RW07_NATIVE_RESOURCE_EP_ORIGIN");
   }
   M0NativeResourceBindingV3 pending;
   ORT_RETURN_IF_NOT(query(3,ep,handle,required_bytes,&pending)==1&&pending.abi==3&&pending.structure_bytes==sizeof(pending)&&
     pending.ep==reinterpret_cast<uintptr_t>(ep)&&pending.context&&pending.device&&pending.queue&&pending.handle==handle,
     "RW07_NATIVE_RESOURCE_BINDING");
   ORT_RETURN_IF_NOT(M0FileKeyV1(canonical)==m0_module_lease_key_,"RW07_NATIVE_RESOURCE_INODE_CHANGED");
   binding=pending;return Status::OK();
  }();
 }
 ORT_CATCH(const std::exception& error){
  ORT_HANDLE_EXCEPTION([&](){status=ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,"RW07_NATIVE_RESOURCE_QUERY: ",error.what());});
 }
 return status;
#else
 ORT_UNUSED_PARAMETER(ep);ORT_UNUSED_PARAMETER(handle);ORT_UNUSED_PARAMETER(required_bytes);ORT_UNUSED_PARAMETER(binding);
 return ORT_MAKE_STATUS(ONNXRUNTIME,FAIL,"RW07_NATIVE_RESOURCE_UNSUPPORTED");
#endif
}
}  // namespace onnxruntime
