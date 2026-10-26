// Optional observation only; never grants Run or tensor access authority.
#pragma once
#include <memory>
#include "core/common/profiler.h"
namespace onnxruntime {
// Shared by the plugin callback adapter and its strict lifecycle test. Keep
// private ownership and observation options together across the ABI boundary.
inline constexpr const char* kM0NativeRunCallbackKeysV1[]={
  "m0.fullgpu.extent_ownership_v2", "m0.fullgpu.bounded_submissions_v1",
  "m0.fullgpu.operator_profile_v1", "m0.fullgpu.actual_session_v2",
  "m0.fullgpu.actual_run_v2", "m0.fullgpu.fault_authorized_v2",
  "m0.fullgpu.diagnostic_fault_mode_v2"};
class M0FirstRunProfileV1 final {
 public:
  template<class Factory> void Start(bool requested,profiling::Profiler& profiler,
      const logging::Logger* logger,const std::string& file,Factory factory){
    if(!requested||attempted_)return;
    attempted_=true;
    ORT_ENFORCE(!file.empty()&&!profiler.IsEnabled(),"RW07_GPU_PROFILE_DESCRIPTOR");
    auto ep=factory();
    if(!ep)return;  // Unsupported EP: do not label CPU events as GPU timing.
    profiler.Initialize(logger);profiler.AddEpProfilers(std::move(ep));
    profiler.StartProfiling(file);active_=true;
    ORT_THROW_IF_ERROR(profiler.GetEpProfilingStatus());
  }
  // Call only after the provider's checked OnRunEnd (also on failure cleanup).
  void Finish(profiling::Profiler& profiler){
    if(!active_)return;
    active_=false;file_=profiler.EndProfiling();
  }
  bool Attempted()const{return attempted_;}
  bool Active()const{return active_;}
  const std::string& File()const{return file_;}
 private:
  bool attempted_=false,active_=false;
  std::string file_;
};
}
