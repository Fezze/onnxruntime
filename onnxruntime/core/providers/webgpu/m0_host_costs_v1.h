// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace onnxruntime::webgpu {
enum class M0HostCostV1:size_t {
  ProgramPrepare,ShaderBuildRequest,PipelineWait,BindGroupCreate,UniformPrepare,
  CompletionFence,QueueCompletionWait,Allocation,MemoryProbe,Upload,Download,
  ErrorScopeWait,DeviceStatusCheck,Count
};
class M0HostCostsV1 final {
 public:
  static constexpr size_t kCount=static_cast<size_t>(M0HostCostV1::Count);
  struct Value{uint64_t calls=0,wall_ns=0;};
  using Snapshot=std::array<Value,kCount>;
  explicit M0HostCostsV1(bool enabled=false):enabled_(enabled){}
  bool Enabled()const{return enabled_;}
  void Enable(bool enabled){enabled_=enabled;}
  void Observe(M0HostCostV1 category,uint64_t nanoseconds)noexcept{
    if(!enabled_)return;
    const auto i=static_cast<size_t>(category);
    calls_[i].fetch_add(1,std::memory_order_relaxed);
    wall_ns_[i].fetch_add(nanoseconds,std::memory_order_relaxed);
  }
  Snapshot Read()const{
    Snapshot result{};
    for(size_t i=0;i<kCount;++i)result[i]={calls_[i].load(std::memory_order_relaxed),wall_ns_[i].load(std::memory_order_relaxed)};
    return result;
  }
  std::string Json()const{
    static constexpr const char* names[]{"program_prepare","shader_build_request","pipeline_wait",
      "bind_group_create","uniform_prepare","completion_fence","queue_completion_wait","allocation",
      "memory_probe","upload","download","error_scope_wait","device_status_check"};
    static_assert(sizeof(names)/sizeof(names[0])==kCount);
    const auto values=Read();std::string result="{";
    for(size_t i=0;i<kCount;++i){
      if(i)result+=",";
      result+='"';result+=names[i];result+="\":{\"calls\":"+std::to_string(values[i].calls)+
          ",\"wall_ns\":"+std::to_string(values[i].wall_ns)+"}";
    }
    return result+"}";
  }
 private:
  bool enabled_=false;
  std::array<std::atomic<uint64_t>,kCount> calls_{};
  std::array<std::atomic<uint64_t>,kCount> wall_ns_{};
};
class M0HostCostTimerV1 final {
 public:
  M0HostCostTimerV1(M0HostCostsV1& costs,M0HostCostV1 category):costs_(costs.Enabled()?&costs:nullptr),category_(category){
    if(costs_)start_=std::chrono::steady_clock::now();
  }
  ~M0HostCostTimerV1(){
    if(costs_)costs_->Observe(category_,uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-start_).count()));
  }
  M0HostCostTimerV1(const M0HostCostTimerV1&)=delete;
  M0HostCostTimerV1& operator=(const M0HostCostTimerV1&)=delete;
 private:
  M0HostCostsV1* costs_;
  M0HostCostV1 category_;
  std::chrono::steady_clock::time_point start_;
};
}  // namespace onnxruntime::webgpu
