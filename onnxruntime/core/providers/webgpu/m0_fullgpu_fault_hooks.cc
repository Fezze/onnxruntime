#include "core/providers/webgpu/m0_fullgpu_fault_hooks.h"
#include <stdexcept>

namespace onnxruntime::webgpu {
namespace {
bool Known(M0FullGpuFaultModeV2 mode) {
  switch (mode) {
    case M0FullGpuFaultModeV2::None:
    case M0FullGpuFaultModeV2::RetainedPair:
    case M0FullGpuFaultModeV2::WrongRole:
    case M0FullGpuFaultModeV2::WrongInstance:
    case M0FullGpuFaultModeV2::CarrierAbaRunPromotion:
    case M0FullGpuFaultModeV2::AfterPushScope:
    case M0FullGpuFaultModeV2::AfterProfilingAttempt:
    case M0FullGpuFaultModeV2::AfterCheckedEnd:
    case M0FullGpuFaultModeV2::FirstTerminalInvalidMap: return true;
  }
  return false;
}
}  // namespace

bool M0FullGpuParseFaultModeV2(std::string_view option, bool certified,
                             M0FullGpuFaultModeV2& out) {
  out = M0FullGpuFaultModeV2::None;
  if (option.empty() || option == "0") return true;
  if (!certified) return false;
  for (const auto mode : {M0FullGpuFaultModeV2::RetainedPair, M0FullGpuFaultModeV2::WrongRole,
                         M0FullGpuFaultModeV2::WrongInstance, M0FullGpuFaultModeV2::CarrierAbaRunPromotion,
                         M0FullGpuFaultModeV2::AfterPushScope, M0FullGpuFaultModeV2::AfterProfilingAttempt,
                         M0FullGpuFaultModeV2::AfterCheckedEnd, M0FullGpuFaultModeV2::FirstTerminalInvalidMap}) {
    const uint32_t value = static_cast<uint32_t>(mode);
    // Closed decimal literals; no allocation or permissive integer parsing.
    const char digits[] = {char('0' + value / 10), char('0' + value % 10)};
    const std::string_view expected = value < 10 ? std::string_view(digits + 1, 1)
                                                : std::string_view(digits, 2);
    if (option == expected) { out = mode; return true; }
  }
  return false;
}

bool M0FullGpuFaultHooksV2::Begin(uint64_t session, uint64_t run,
                                M0FullGpuFaultModeV2 mode, bool certified) {
  if (!session || !run || !Known(mode) || (mode != M0FullGpuFaultModeV2::None && !certified)) return false;
  std::lock_guard<std::mutex> lock{mutex_};
  const auto previous = session_last_run_.find(session);
  if (active_runs_.count(run) || session_active_run_.count(session) ||
      (previous != session_last_run_.end() && run <= previous->second)) return false;
  auto state = std::make_shared<RunState>();
  state->session = session; state->run = run; state->mode = mode;
  // Roll back partial container insertions if host administrative allocation
  // fails. No helper entry may look active after Begin throws.
  try {
    active_runs_.emplace(run, state);
    session_active_run_.emplace(session, run);
    session_last_run_[session] = run;
  } catch (...) {
    active_runs_.erase(run); session_active_run_.erase(session);
    throw;
  }
  return true;
}

bool M0FullGpuFaultHooksV2::End(uint64_t session, uint64_t run) {
  std::lock_guard<std::mutex> lock{mutex_};
  const auto it = active_runs_.find(run);
  if (it == active_runs_.end() || it->second->session != session) return false;
  session_active_run_.erase(session);
  active_runs_.erase(it);  // retained terminal tickets keep immutable Run provenance
  return true;
}

bool M0FullGpuFaultHooksV2::Mode(uint64_t session, uint64_t run, M0FullGpuFaultModeV2& out) const {
  out = M0FullGpuFaultModeV2::None;
  std::lock_guard<std::mutex> lock{mutex_};
  const auto it = active_runs_.find(run);
  if (it == active_runs_.end() || it->second->session != session) return false;
  out = it->second->mode;
  return true;
}

void M0FullGpuFaultHooksV2::ThrowIfRequested(uint64_t session, uint64_t run,
                                          M0FullGpuFaultCheckpointV2 checkpoint) const {
  M0FullGpuFaultModeV2 mode;
  if (!Mode(session, run, mode)) throw std::logic_error("RW07 diagnostic fault Run identity unavailable");
  switch (checkpoint) {
    case M0FullGpuFaultCheckpointV2::AfterPushScope:
      if (mode == M0FullGpuFaultModeV2::AfterPushScope) throw std::runtime_error("RW07 injected after pushed error scope");
      return;
    case M0FullGpuFaultCheckpointV2::AfterProfilingAttempt:
      if (mode == M0FullGpuFaultModeV2::AfterProfilingAttempt) throw std::runtime_error("RW07 injected after profiling attempt");
      return;
    case M0FullGpuFaultCheckpointV2::AfterCheckedEnd:
      if (mode == M0FullGpuFaultModeV2::AfterCheckedEnd) throw std::runtime_error("RW07 injected after checked queue end");
      return;
  }
  throw std::invalid_argument("RW07 unknown diagnostic fault checkpoint");
}

bool M0FullGpuFaultHooksV2::RegisterTerminal(uintptr_t handle, uint64_t generation,
                                           uint64_t session, uint64_t run) {
  if (!handle || !generation) return false;
  std::lock_guard<std::mutex> lock{mutex_};
  const auto it = active_runs_.find(run);
  if (it == active_runs_.end() || it->second->session != session) return false;
  if (it->second->mode != M0FullGpuFaultModeV2::FirstTerminalInvalidMap) return true;
  const auto existing = terminals_.find(generation);
  if (existing != terminals_.end()) {
    return existing->second.handle == handle && existing->second.state == it->second;
  }
  terminals_.emplace(generation, Terminal{handle, it->second});
  return true;
}

bool M0FullGpuFaultHooksV2::ConsumeInvalidTerminalMap(uintptr_t handle, uint64_t generation) {
  std::lock_guard<std::mutex> lock{mutex_};
  const auto it = terminals_.find(generation);
  if (it == terminals_.end() || it->second.handle != handle) return false;
  bool expected = false;
  return it->second.state->map_consumed.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
}

bool M0FullGpuFaultHooksV2::RetireTerminal(uintptr_t handle, uint64_t generation) {
  std::lock_guard<std::mutex> lock{mutex_};
  const auto it = terminals_.find(generation);
  if (it == terminals_.end() || it->second.handle != handle) return false;
  terminals_.erase(it);
  return true;
}
}  // namespace onnxruntime::webgpu
