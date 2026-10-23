// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <charconv>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <string_view>
#include "core/common/common.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {
inline constexpr uint64_t kM0MemoryReserveBytesV1 = 512ull * 1024 * 1024;
#if defined(ORT_M0_ENFORCE_MEMORY_RESERVE)
inline constexpr bool kM0EnforceMemoryReserveV1 = true;
#else
inline constexpr bool kM0EnforceMemoryReserveV1 = false;
#endif
inline bool M0MemoryHeadroomV1(uint64_t available, uint64_t additional) {
  return available >= kM0MemoryReserveBytesV1 && additional <= available - kM0MemoryReserveBytesV1;
}
// Select a planning target, never an execution permission in monitor mode.
inline uint64_t M0MemoryPlanningBudgetV1(uint64_t available, uint64_t future, uint64_t minimum,
                                       bool enforce = kM0EnforceMemoryReserveV1) {
  const uint64_t headroom = available >= kM0MemoryReserveBytesV1 &&
      future <= available - kM0MemoryReserveBytesV1 ? available - kM0MemoryReserveBytesV1 - future : 0;
  if (enforce || minimum <= headroom / 2) return headroom / 2;
  if (minimum <= headroom) return headroom;
  const uint64_t remaining = available >= future ? available - future : 0;
  if (minimum <= remaining / 2) return remaining / 2;
  if (minimum <= remaining) return remaining;
  return std::numeric_limits<uint64_t>::max();
}

// Output is assigned only after the complete bounded reading is validated.
inline common::Status M0ReadMemoryAvailableV1(uint64_t& output) {
#if defined(__ANDROID__)
  std::array<char, 16384> buffer;
  auto* file = std::fopen("/proc/meminfo", "r");
  ORT_RETURN_IF_NOT(file, "RW07_NATIVE_MEMORY_UNAVAILABLE");
  const size_t count = std::fread(buffer.data(), 1, buffer.size(), file);
  const bool complete = std::feof(file) && !std::ferror(file);
  std::fclose(file);
  ORT_RETURN_IF_NOT(complete, "RW07_NATIVE_MEMORY_UNAVAILABLE");
  std::string_view remaining(buffer.data(), count);
  bool found = false;
  uint64_t available_kib = 0;
  while (!remaining.empty()) {
    const auto end = remaining.find('\n');
    auto line = remaining.substr(0, end);
    remaining = end == std::string_view::npos ? std::string_view{} : remaining.substr(end + 1);
    constexpr std::string_view kField = "MemAvailable:";
    if (line.substr(0, kField.size()) != kField) continue;
    ORT_RETURN_IF_NOT(!found, "RW07_NATIVE_MEMORY_DUPLICATE_AVAILABLE");
    found = true;
    line.remove_prefix(kField.size());
    const auto first = line.find_first_not_of(" \t");
    ORT_RETURN_IF_NOT(first != std::string_view::npos, "RW07_NATIVE_MEMORY_INVALID_AVAILABLE");
    line.remove_prefix(first);
    const auto parsed = std::from_chars(line.data(), line.data() + line.size(), available_kib);
    ORT_RETURN_IF_NOT(parsed.ec == std::errc{}, "RW07_NATIVE_MEMORY_INVALID_AVAILABLE");
    line.remove_prefix(static_cast<size_t>(parsed.ptr - line.data()));
    ORT_RETURN_IF_NOT(!line.empty() && (line.front() == ' ' || line.front() == '\t'),
                      "RW07_NATIVE_MEMORY_INVALID_UNIT");
    const auto unit = line.find_first_not_of(" \t");
    ORT_RETURN_IF_NOT(unit != std::string_view::npos, "RW07_NATIVE_MEMORY_INVALID_UNIT");
    line.remove_prefix(unit);
    ORT_RETURN_IF_NOT(line.substr(0, 2) == "kB", "RW07_NATIVE_MEMORY_INVALID_UNIT");
    line.remove_prefix(2);
    ORT_RETURN_IF_NOT(line.find_first_not_of(" \t\r") == std::string_view::npos,
                      "RW07_NATIVE_MEMORY_INVALID_AVAILABLE");
  }
  ORT_RETURN_IF_NOT(found && available_kib <= std::numeric_limits<uint64_t>::max() / 1024,
                    "RW07_NATIVE_MEMORY_UNAVAILABLE");
  output = available_kib * 1024;
#else
  output = std::numeric_limits<uint64_t>::max();
#endif
  return common::Status::OK();
}

// Android shares physical memory with the GPU. This is a current availability
// observation before allocation, not GPU residency or an OS reservation.
inline common::Status M0CheckMemoryReserveV1(uint64_t additional_bytes) {
#if defined(__ANDROID__)
  uint64_t available_bytes = 0;
  const auto reading = M0ReadMemoryAvailableV1(available_bytes);
  if (!reading.IsOK()) {
    LOGS_DEFAULT(WARNING) << "M0_RW07_MEMORY_WARNING available=unknown policy="
                         << (kM0EnforceMemoryReserveV1 ? "enforce" : "monitor");
    return kM0EnforceMemoryReserveV1 ? reading : common::Status::OK();
  }
  LOGS_DEFAULT(INFO) << "M0_RW07_MEMORY_OBSERVATION additional_bytes=" << additional_bytes
                     << " available_bytes=" << available_bytes << " reserve_bytes=" << kM0MemoryReserveBytesV1;
  if (!M0MemoryHeadroomV1(available_bytes, additional_bytes)) {
    LOGS_DEFAULT(WARNING) << "M0_RW07_MEMORY_WARNING additional_bytes=" << additional_bytes
                         << " available_bytes=" << available_bytes << " reserve_bytes=" << kM0MemoryReserveBytesV1
                         << " policy=" << (kM0EnforceMemoryReserveV1 ? "enforce" : "monitor");
  }
  ORT_RETURN_IF_NOT(!kM0EnforceMemoryReserveV1 || M0MemoryHeadroomV1(available_bytes, additional_bytes),
                    "RW07_NATIVE_MEMORY_RESERVE additional_bytes=", additional_bytes,
                    " available_bytes=", available_bytes, " reserve_bytes=", kM0MemoryReserveBytesV1);
#else
  (void)additional_bytes;
#endif
  return common::Status::OK();
}
}  // namespace onnxruntime
