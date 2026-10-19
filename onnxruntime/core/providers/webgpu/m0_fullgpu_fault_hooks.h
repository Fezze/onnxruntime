// Source-private diagnostic administration only. Never reads tensor payloads.
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace onnxruntime::webgpu {
enum class M0FullGpuFaultModeV2 : uint32_t {
  None = 0, RetainedPair = 1, WrongRole = 2, WrongInstance = 3,
  CarrierAbaRunPromotion = 4, AfterPushScope = 10,
  AfterProfilingAttempt = 11, AfterCheckedEnd = 12,
  FirstTerminalInvalidMap = 20,
  DestroyDeviceAtEnd = 21,
  ZeroDeadlineAtEnd = 22,
};
enum class M0FullGpuFaultCheckpointV2 : uint32_t {
  AfterPushScope = 10, AfterProfilingAttempt = 11, AfterCheckedEnd = 12,
};

// Only a certified diagnostic Session may request nonzero modes. Unknown and
// noncanonical strings fail closed; this is not model shape/control parsing.
bool M0FullGpuParseFaultModeV2(std::string_view option, bool certified_diagnostic_session,
                             M0FullGpuFaultModeV2& out);

// May be stored in a context, but all modes/tickets belong to an actual unique
// Session+Run. No thread-local/global selector or model values are present.
class M0FullGpuFaultHooksV2 final {
 public:
  bool Begin(uint64_t session, uint64_t run, M0FullGpuFaultModeV2 mode,
             bool certified_diagnostic_session);
  bool End(uint64_t session, uint64_t run);
  bool Mode(uint64_t session, uint64_t run, M0FullGpuFaultModeV2& out) const;
  void ThrowIfRequested(uint64_t session, uint64_t run,
                        M0FullGpuFaultCheckpointV2 checkpoint) const;
  // Register only actual administrative terminal outputs, never arbitrary
  // model/intermediate resources. Registration must precede End/Run return.
  bool RegisterTerminal(uintptr_t handle, uint64_t generation, uint64_t session, uint64_t run);
  // True once per mode20 Run, including after End while terminal is retained.
  // Root must perform a REAL invalid MapAsync range; this returns no fake status.
  bool ConsumeInvalidTerminalMap(uintptr_t handle, uint64_t generation);
  bool RetireTerminal(uintptr_t handle, uint64_t generation);

 private:
  struct RunState {
    uint64_t session = 0, run = 0;
    M0FullGpuFaultModeV2 mode = M0FullGpuFaultModeV2::None;
    std::atomic<bool> map_consumed{false};
  };
  struct Terminal {
    uintptr_t handle = 0;
    std::shared_ptr<RunState> state;
  };
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<RunState>> active_runs_;
  std::unordered_map<uint64_t, uint64_t> session_last_run_;
  std::unordered_map<uint64_t, uint64_t> session_active_run_;
  std::unordered_map<uint64_t, Terminal> terminals_;  // process-unique generation key
};
}  // namespace onnxruntime::webgpu
