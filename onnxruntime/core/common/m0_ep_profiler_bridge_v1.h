// C API bridge for retained EP profilers whose per-ORT-event hooks are unused.
// No GPU-specific types or device assumptions; the EP retains the profiler.
#pragma once
#include <chrono>
#include <vector>
#include "core/common/profiler_common.h"
#include "core/session/onnxruntime_cxx_api.h"
namespace onnxruntime {
class M0EpProfilerBridgeV1 final:public OrtEpProfilerImpl {
 public:
  explicit M0EpProfilerBridgeV1(profiling::EpProfiler& profiler):OrtEpProfilerImpl{},profiler_(profiler){
    ort_version_supported=ORT_API_VERSION;Release=ReleaseImpl;
    StartProfiling=StartImpl;EndProfiling=EndImpl;
    StartEvent=nullptr;StopEvent=nullptr;
  }
 private:
  static void ORT_API_CALL ReleaseImpl(OrtEpProfilerImpl* self)noexcept{delete static_cast<M0EpProfilerBridgeV1*>(self);}
  static OrtStatus* Error(const char* message)noexcept{return Ort::GetApi().CreateStatus(ORT_FAIL,message);}
  static OrtStatus* ORT_API_CALL StartImpl(OrtEpProfilerImpl* ptr,int64_t offset_ns)noexcept{
    try{
      if(offset_ns<0)return Error("Negative EP profiling offset");
      auto& self=*static_cast<M0EpProfilerBridgeV1*>(ptr);
      self.start_=std::chrono::high_resolution_clock::now()-std::chrono::nanoseconds(offset_ns);
      const auto status=self.profiler_.StartProfiling(self.start_);
      return status.IsOK()?nullptr:Error(status.ErrorMessage().c_str());
    }catch(const std::exception& e){return Error(e.what());}catch(...){return Error("EP profiler start failed");}
  }
  static OrtStatus* ORT_API_CALL EndImpl(OrtEpProfilerImpl* ptr,OrtProfilingEventsContainer* container)noexcept{
    try{
      if(!container)return Error("EP profiling container absent");
      auto& self=*static_cast<M0EpProfilerBridgeV1*>(ptr);profiling::Events records;
      self.profiler_.EndProfiling(self.start_,records);
      std::vector<Ort::ProfilingEvent> batch;batch.reserve(256);
      for(const auto& record:records){
        std::vector<const char*> keys,values;keys.reserve(record.args.size());values.reserve(record.args.size());
        for(const auto& arg:record.args){keys.push_back(arg.first.c_str());values.push_back(arg.second.c_str());}
        batch.emplace_back(static_cast<OrtProfilingEventCategory>(record.cat),record.pid,record.tid,
          record.name.c_str(),record.ts,record.dur,keys.data(),values.data(),keys.size());
        if(batch.size()==256){auto status=Ort::UnownedProfilingEventsContainer(container).AddEvents(batch);
          if(!status.IsOK())return status.release();batch.clear();}
      }
      if(!batch.empty()){auto status=Ort::UnownedProfilingEventsContainer(container).AddEvents(batch);if(!status.IsOK())return status.release();}
      return nullptr;
    }catch(const std::exception& e){return Error(e.what());}catch(...){return Error("EP profiler collection failed");}
  }
  profiling::EpProfiler& profiler_;
  TimePoint start_{};
};
static_assert(int(profiling::API_EVENT)==int(OrtProfilingEventCategory_API),"Profiling category ABI");
}
