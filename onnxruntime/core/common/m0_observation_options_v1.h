// Optional telemetry only. Set before loading/initializing the runtime.
#pragma once
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <initializer_list>
namespace onnxruntime {
inline int64_t M0MonotonicMicrosecondsV1() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline bool M0ObservationOptionV1(const char* name) {
  const char* value = std::getenv(name);
  return !value || std::strcmp(value, "0") != 0;
}
inline bool M0MemoryObservationsV1() {
  static const bool enabled = M0ObservationOptionV1("M0_MEMORY_OBSERVATIONS");
  return enabled;
}
inline bool M0WorkloadObservationsV1() {
  static const bool enabled = M0ObservationOptionV1("M0_WORKLOAD_OBSERVATIONS");
  return enabled;
}
inline bool M0BatchHeavyProgramsV1() {
  static const bool enabled = [](){const char* value=std::getenv("M0_BATCH_HEAVY_PROGRAMS");return value&&std::strcmp(value,"1")==0;}();
  return enabled;
}
inline uint32_t M0NativeDispatchWindowV1(bool batched = M0BatchHeavyProgramsV1()) {
  return batched ? 128u : 16u;
}
// Retain the existing qualified heavy-dispatch product envelope. Actual
// algorithm work determines element counts; the aggregate queue limit is separate.
inline constexpr uint32_t kM0HeavyProductsPerDispatchV1=1u<<22;
inline uint32_t M0HeavyProductsPerDispatchV1(bool batched=M0BatchHeavyProgramsV1()){
  return batched?8u*kM0HeavyProductsPerDispatchV1:kM0HeavyProductsPerDispatchV1;
}
inline constexpr uint64_t kM0ResidualPackedScratchBytesV1=4ull*1024*1024;
inline constexpr uint32_t kM0ResidualPackedWeightFlagWordsV1=16;
inline constexpr uint64_t kM0ResidualRegisterWeightScratchBytesV1=4ull*1024*1024;
inline constexpr uint64_t kM0ResidualPackedFlagWordsV1=(kM0ResidualPackedScratchBytesV1/4+255)/256+64+kM0ResidualPackedWeightFlagWordsV1;
inline constexpr uint64_t M0NativeResidualFixedScratchBytesV1(uint32_t channels,bool packed){
  return uint64_t(channels)*6*sizeof(float)+15+
    (packed?kM0ResidualPackedScratchBytesV1+kM0ResidualRegisterWeightScratchBytesV1+kM0ResidualPackedFlagWordsV1*sizeof(uint32_t)+3*15+3*sizeof(uint32_t)+15:0);
}
inline constexpr uint64_t kM0NativeUniformBytesV1=512;
inline constexpr uint64_t kM0UniformArenaBytesV1=64*1024;
inline uint64_t M0NativePendingUniformBytesV1(bool batched=M0BatchHeavyProgramsV1()){
  // A recording retains at most one sealed and one current block; preparation
  // may own one more. A full current block is submitted before another is allocated.
  return batched?3*kM0UniformArenaBytesV1:(M0NativeDispatchWindowV1(false)+1)*kM0NativeUniformBytesV1;
}
inline bool M0VectorizedConvV1() {
  static const bool enabled = [](){const char* value=std::getenv("M0_VECTORIZED_CONVOLUTION");return value&&std::strcmp(value,"1")==0;}();
  return enabled;
}
inline bool M0ParallelNormV1() {
  static const bool enabled = [](){const char* value=std::getenv("M0_PARALLEL_NORMALIZATION");return value&&std::strcmp(value,"1")==0;}();
  return enabled;
}
inline bool M0FusedNormSnakeV1() {
  static const bool enabled = [](){const char* value=std::getenv("M0_FUSED_NORMALIZATION_SNAKE");return value&&std::strcmp(value,"1")==0;}();
  return enabled;
}
inline bool M0TiledConvV1() {
  static const bool enabled = [](){const char* value=std::getenv("M0_TILED_CONVOLUTION");return value&&std::strcmp(value,"1")==0;}();
  return enabled;
}
inline bool M0LstmInputProjectionV1() {
  static const bool enabled = [](){const char* value=std::getenv("M0_LSTM_INPUT_PROJECTION");return value&&std::strcmp(value,"1")==0;}();
  return enabled;
}
inline bool M0FusedRecurrenceMomentsV1() {
  static const bool enabled = [](){const char* value=std::getenv("M0_FUSED_RECURRENCE_MOMENTS");return value&&std::strcmp(value,"1")==0;}();
  return enabled;
}
inline bool M0CoalescedRecurrenceWeightsV1() {
  static const bool enabled = [](){const char* value=std::getenv("M0_COALESCED_RECURRENCE_WEIGHTS");return value&&std::strcmp(value,"1")==0;}();
  return enabled;
}
inline bool M0OutputChannelTiledConvV1() {
  static const bool enabled=[](){const char* value=std::getenv("M0_OUTPUT_CHANNEL_TILED_CONVOLUTION");return value&&std::strcmp(value,"1")==0;}();
  return enabled;
}
inline bool M0GpuNumericTraceV1() {
  static const bool enabled = []() {
    const char* value = std::getenv("M0_GPU_NUMERIC_TRACE");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument("Invalid M0_GPU_NUMERIC_TRACE");
  }();
  return enabled;
}
inline bool M0GpuOperatorProfileV1() {
  static const bool enabled=[](){const char* value=std::getenv("M0_GPU_OPERATOR_PROFILE");return value&&std::strcmp(value,"1")==0;}();
  return enabled;
}
inline const char* M0GpuValidationV1() {
  static const char* mode = []() {
    const char* configured = std::getenv("M0_GPU_VALIDATION");
    if (!configured) return "full";
    for (const char* supported : {"full", "basic", "wgpuOnly", "disabled"}) {
      if (std::strcmp(configured, supported) == 0) return supported;
    }
    throw std::invalid_argument("Invalid M0_GPU_VALIDATION");
  }();
  return mode;
}
}  // namespace onnxruntime
