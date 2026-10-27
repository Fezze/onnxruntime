// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#if defined(__linux__)
#include <array>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <dlfcn.h>

namespace onnxruntime::ep::adapter::m0_detail {
struct M0CoreModuleOpsV1 {
  static int Address(const void* value,Dl_info* info) noexcept {return dladdr(value,info);}
  static void* Open(const char* path,int flags) noexcept {return dlopen(path,flags);}
  static void* Symbol(void* module,const char* name) noexcept {return dlsym(module,name);}
  static void Close(void* module) noexcept {(void)dlclose(module);}
};

// One lease per retained OpKernel, never a process-global pointer cache.
// The live API anchor and each query must belong to the same loaded image.
// Only addresses are retained: queries still read the current run/cancel state.
// The small operations seam lets unit tests exercise foreign/missing modules
// and prove that repeated/concurrent cancellation checks do not reopen them.
// Default operations are the real platform dynamic loader.
template<class ModuleOps=M0CoreModuleOpsV1>
class M0CoreContextQueryLeaseV1 final {
 public:
  M0CoreContextQueryLeaseV1()=default;
  M0CoreContextQueryLeaseV1(const M0CoreContextQueryLeaseV1&)=delete;
  M0CoreContextQueryLeaseV1& operator=(const M0CoreContextQueryLeaseV1&)=delete;
  ~M0CoreContextQueryLeaseV1(){if(module_)ModuleOps::Close(module_);}

  void* Lookup(const void* api_function,const char* symbol) const noexcept {
    if(!api_function||!symbol)return nullptr;
    size_t index=0;
    for(;index<kNames.size();++index)if(std::strcmp(symbol,kNames[index])==0)break;
    if(index==kNames.size())return nullptr;
    try {
      std::call_once(initialized_,[this,api_function](){Bind(api_function);});
    } catch(...) {return nullptr;}
    // An API replacement is not authorized by a previous module's lease.
    return api_function==api_function_?queries_[index]:nullptr;
  }

 private:
  void Bind(const void* api_function) const noexcept {
    Dl_info api_info{};
    if(!ModuleOps::Address(api_function,&api_info)||!api_info.dli_fname||!api_info.dli_fbase)return;
    api_function_=api_function;
    module_=ModuleOps::Open(api_info.dli_fname,RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
    // Android qualification owns a PIE core in the main executable. The same
    // exact base comparison below applies to this fallback handle as to a DSO.
    if(!module_)module_=ModuleOps::Open(nullptr,RTLD_NOW|RTLD_LOCAL);
    if(!module_)return;
    for(size_t index=0;index<kNames.size();++index){
      void* query=ModuleOps::Symbol(module_,kNames[index]);Dl_info query_info{};
      if(query&&ModuleOps::Address(query,&query_info)&&query_info.dli_fbase==api_info.dli_fbase)
        queries_[index]=query;
    }
  }
  static constexpr std::array<const char*,3> kNames{
    "OrtM0FullGpuExtentOwnerEpochV1","OrtM0FullGpuExtentRunIdV2","OrtM0FullGpuReadCancellationV1"};
  mutable std::once_flag initialized_;
  mutable const void* api_function_=nullptr;
  mutable void* module_=nullptr;
  mutable std::array<void*,3> queries_{};
};
}  // namespace onnxruntime::ep::adapter::m0_detail
#endif  // __linux__
