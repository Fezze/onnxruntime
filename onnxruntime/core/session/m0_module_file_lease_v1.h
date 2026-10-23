// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include "core/common/common.h"
#include "sha256.h"

namespace onnxruntime {
class M0ModuleFileLeaseV1 final {
 public:
  explicit M0ModuleFileLeaseV1(std::string_view expected_digest, bool deep_audit = false)
      : expected_digest_(expected_digest), deep_audit_(deep_audit) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(M0ModuleFileLeaseV1);
  int fd = -1;
  std::filesystem::path path;
  struct stat stamp{};
  ~M0ModuleFileLeaseV1() {
    if (fd >= 0) close(fd);
  }
  bool DeepAuditEnabled() const { return deep_audit_; }
  bool AuditRegistry(uintptr_t handle) const { return handle == 0 || deep_audit_; }

  // The first observation always authenticates bytes. Later observations keep
  // inode/timestamp checks; diagnostic mode also rereads and hashes the file.
  // Return whether bytes were hashed so tests can verify the selected path.
  bool Verify(bool force_digest = false) const {
    struct stat current{}, named{};
    ORT_ENFORCE(fd >= 0 && fstat(fd, &current) == 0 && stat(path.c_str(), &named) == 0,
                "RW07_LSTM_MODULE_FILE_STAT");
    ORT_ENFORCE(S_ISREG(current.st_mode) && current.st_dev == stamp.st_dev && current.st_ino == stamp.st_ino &&
                    named.st_dev == stamp.st_dev && named.st_ino == stamp.st_ino && current.st_size == stamp.st_size &&
                    current.st_mtim.tv_sec == stamp.st_mtim.tv_sec && current.st_mtim.tv_nsec == stamp.st_mtim.tv_nsec &&
                    current.st_ctim.tv_sec == stamp.st_ctim.tv_sec && current.st_ctim.tv_nsec == stamp.st_ctim.tv_nsec,
                "RW07_LSTM_MODULE_FILE_CHANGED");
    if (authenticated_.load(std::memory_order_relaxed) && !force_digest && !deep_audit_) return false;
    model_package::Sha256 hash;
    std::array<char, 65536> bytes;
    off_t offset = 0;
    while (offset < current.st_size) {
      ssize_t count = pread(fd, bytes.data(), bytes.size(), offset);
      if (count < 0 && errno == EINTR) continue;
      ORT_ENFORCE(count > 0, "RW07_LSTM_MODULE_FILE_READ");
      hash.Update(bytes.data(), static_cast<size_t>(count));
      offset += count;
    }
    ORT_ENFORCE(hash.FinalHex() == expected_digest_, "RW07_LSTM_MODULE_DIGEST");
    struct stat after{};
    ORT_ENFORCE(fstat(fd, &after) == 0 && after.st_size == current.st_size &&
                    after.st_mtim.tv_sec == current.st_mtim.tv_sec && after.st_mtim.tv_nsec == current.st_mtim.tv_nsec &&
                    after.st_ctim.tv_sec == current.st_ctim.tv_sec && after.st_ctim.tv_nsec == current.st_ctim.tv_nsec,
                "RW07_LSTM_MODULE_HASH_RACE");
    authenticated_.store(true, std::memory_order_relaxed);
    return true;
  }

 private:
  const std::string expected_digest_;
  const bool deep_audit_;
  mutable std::atomic<bool> authenticated_{false};
};
}  // namespace onnxruntime
