// Source-private RW07 administrative identity. No tensor values are inspected.
#pragma once
#include <atomic>
#include <cstdint>

namespace onnxruntime {

// A strict extent Session permits exactly one active Run. Ordinary Sessions
// must not call this opt-in helper: their existing concurrency stays intact.
// The coordinator must integrate a scope spanning ExecuteGraph and provider
// completion, including all exception/early-return paths, before certification.
class M0FullGpuRunOwnership final {
 public:
  M0FullGpuRunOwnership() = default;
  M0FullGpuRunOwnership(const M0FullGpuRunOwnership&) = delete;
  M0FullGpuRunOwnership& operator=(const M0FullGpuRunOwnership&) = delete;

  // Zero means an active Run already owns this Session. IDs are process-wide,
  // monotonic, never reused, and independent of caller tags or thread IDs.
  uint64_t Begin();
  // A stale or foreign ID cannot close a different live Run.
  bool End(uint64_t id);
  uint64_t Active() const { return active_.load(std::memory_order_acquire); }
  void Poison() { poisoned_.store(true, std::memory_order_release); }
  bool Poisoned() const { return poisoned_.load(std::memory_order_acquire); }

 private:
  std::atomic<uint64_t> active_{0};
  std::atomic<bool> poisoned_{false};
};

// Scope destruction retires only the Run this scope actually acquired.
class M0FullGpuRunScope final {
 public:
  explicit M0FullGpuRunScope(M0FullGpuRunOwnership& owner)
      : owner_{owner}, id_{owner.Begin()} {}
  ~M0FullGpuRunScope() { if (id_) owner_.End(id_); }
  M0FullGpuRunScope(const M0FullGpuRunScope&) = delete;
  M0FullGpuRunScope& operator=(const M0FullGpuRunScope&) = delete;
  uint64_t Id() const { return id_; }

 private:
  M0FullGpuRunOwnership& owner_;
  const uint64_t id_;
};

}  // namespace onnxruntime
