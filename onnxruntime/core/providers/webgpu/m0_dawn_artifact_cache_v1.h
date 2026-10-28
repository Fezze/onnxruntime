// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace onnxruntime::webgpu {
// Stores Dawn's opaque compiler/pipeline artifacts, never model tensors or PCM.
// Dawn supplies the shader/compiler/options/device/driver cache key. This layer
// adds bounded private storage, atomic replacement and key/payload integrity.
class M0DawnArtifactCacheV1 final {
 public:
  struct Statistics {
    uint64_t hits=0, misses=0, stores=0, corrupt=0, evicted=0, bytes_loaded=0;
  };
  M0DawnArtifactCacheV1(std::string directory,uint64_t quota_bytes);
  ~M0DawnArtifactCacheV1();
  M0DawnArtifactCacheV1(const M0DawnArtifactCacheV1&)=delete;
  M0DawnArtifactCacheV1& operator=(const M0DawnArtifactCacheV1&)=delete;
  static std::unique_ptr<M0DawnArtifactCacheV1> FromEnvironment() noexcept;
  static M0DawnArtifactCacheV1* ProcessCache() noexcept;
  bool Available()const noexcept{return directory_fd_>=0&&lock_fd_>=0;}
  size_t Load(size_t key_size,const uint8_t* key,size_t value_size,uint8_t* value) noexcept;
  void Store(size_t key_size,const uint8_t* key,size_t value_size,const uint8_t* value) noexcept;
  Statistics Snapshot()const;

 private:
  int directory_fd_=-1,lock_fd_=-1;
  uint64_t quota_bytes_=0,temporary_serial_=0;
  mutable std::mutex mutex_;
  Statistics statistics_;
};
}  // namespace onnxruntime::webgpu
