#include "core/framework/m0_fullgpu_run_ownership.h"
#include <limits>
#include <stdexcept>

namespace onnxruntime {
namespace {
std::atomic<uint64_t> next_run{1};
uint64_t NextRun() {
  uint64_t value = next_run.load(std::memory_order_relaxed);
  for (;;) {
    // Fail before wrapping, rather than reissuing an old identity.
    if (value == std::numeric_limits<uint64_t>::max()) {
      throw std::overflow_error("RW07 Run identity exhausted");
    }
    if (next_run.compare_exchange_weak(value, value + 1,
                                       std::memory_order_relaxed)) {
      return value;
    }
  }
}
}  // namespace

uint64_t M0FullGpuRunOwnership::Begin() {
  if (Poisoned()) return 0;
  const uint64_t id = NextRun();
  uint64_t expected = 0;
  if (!active_.compare_exchange_strong(expected, id, std::memory_order_acq_rel)) return 0;
  if (Poisoned()) { End(id); return 0; }
  return id;
}

bool M0FullGpuRunOwnership::End(uint64_t id) {
  if (!id) return false;
  return active_.compare_exchange_strong(id, 0, std::memory_order_acq_rel);
}
}  // namespace onnxruntime
