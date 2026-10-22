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
// Android shares physical memory with the GPU. This is a current availability
// check before allocation, not a measurement of GPU residency or an OS reservation.
inline common::Status M0CheckMemoryReserveV1(uint64_t additional_bytes) {
#if defined(__ANDROID__)
  constexpr uint64_t kReserveBytes = 512ull * 1024 * 1024;
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
  const uint64_t available_bytes = available_kib * 1024;
  LOGS_DEFAULT(INFO) << "M0_RW07_MEMORY_ADMISSION additional_bytes=" << additional_bytes
                     << " available_bytes=" << available_bytes << " reserve_bytes=" << kReserveBytes;
  ORT_RETURN_IF_NOT(available_bytes >= kReserveBytes && additional_bytes <= available_bytes - kReserveBytes,
                    "RW07_NATIVE_MEMORY_RESERVE additional_bytes=", additional_bytes,
                    " available_bytes=", available_bytes, " reserve_bytes=", kReserveBytes);
#else
  (void)additional_bytes;
#endif
  return common::Status::OK();
}
}  // namespace onnxruntime
