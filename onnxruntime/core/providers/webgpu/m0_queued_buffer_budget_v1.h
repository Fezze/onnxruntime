// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace onnxruntime::webgpu {
// Keys identify backing buffers already retained by command bind groups. Repeated
// references retain one backing allocation, not another copy of its byte extent.
class M0QueuedBufferBudgetV1 {
 public:
  using Reference = std::pair<uintptr_t, uint64_t>;
  M0QueuedBufferBudgetV1() = default;
  M0QueuedBufferBudgetV1(const M0QueuedBufferBudgetV1&) = delete;
  M0QueuedBufferBudgetV1& operator=(const M0QueuedBufferBudgetV1&) = delete;
  M0QueuedBufferBudgetV1(M0QueuedBufferBudgetV1&&) = delete;
  M0QueuedBufferBudgetV1& operator=(M0QueuedBufferBudgetV1&&) = delete;
  uint64_t Bytes() const { return bytes_; }
  bool Empty() const { return references_.empty(); }
  uint64_t Additional(const std::vector<Reference>& references) const {
    uint64_t added = 0;
    for (std::size_t i = 0; i < references.size(); ++i) {
      const auto& [key, size] = references[i];
      if (key == 0 || size == 0) throw std::invalid_argument("RW07 queued buffer identity/extent");
      const auto found = Find(key);
      if (found != references_.end() && found->first == key) {
        if (found->second != size) throw std::invalid_argument("RW07 queued buffer extent changed");
        continue;
      }
      bool duplicate = false;
      for (std::size_t j = 0; j < i; ++j) {
        if (references[j].first == key) {
          if (references[j].second != size) throw std::invalid_argument("RW07 duplicate buffer extent changed");
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        if (size > std::numeric_limits<uint64_t>::max() - added) throw std::overflow_error("RW07 queued buffer sum");
        added += size;
      }
    }
    if (added > std::numeric_limits<uint64_t>::max() - bytes_) throw std::overflow_error("RW07 queued buffer total");
    return added;
  }
  bool NeedsDrain(uint64_t budget, const std::vector<Reference>& references,
                  uint64_t separately_retained_bytes = 0) const {
    const auto added = Additional(references);
    if (separately_retained_bytes > std::numeric_limits<uint64_t>::max() - bytes_)
      throw std::overflow_error("RW07 queued transfer and buffer total");
    const auto retained = bytes_ + separately_retained_bytes;
    return retained != 0 && (retained >= budget || added > budget - retained);
  }
  void Track(const std::vector<Reference>& references) {
    const auto added = Additional(references);
    for (const auto& value : references) {
      const auto found = Find(value.first);
      if (found == references_.end() || found->first != value.first) references_.insert(found, value);
    }
    bytes_ += added;
  }
  void Reset() { references_.clear(); bytes_ = 0; }
 private:
  std::vector<Reference>::const_iterator Find(uintptr_t key) const {
    return std::lower_bound(references_.begin(), references_.end(), key,
                            [](const Reference& value, uintptr_t wanted) { return value.first < wanted; });
  }
  std::vector<Reference>::iterator Find(uintptr_t key) {
    return std::lower_bound(references_.begin(), references_.end(), key,
                            [](const Reference& value, uintptr_t wanted) { return value.first < wanted; });
  }
  std::vector<Reference> references_;
  uint64_t bytes_ = 0;
};
// CPU bytes for one immutable uniform upload. A sealed block can remain owned
// by a command that was still being prepared when an earlier queue was drained.
class M0UniformPackingV1 {
 public:
  static constexpr size_t kMaximumBytes=64*1024;
  explicit M0UniformPackingV1(size_t capacity):capacity_(capacity){
    if(!capacity||capacity>kMaximumBytes||capacity%16)throw std::invalid_argument("RW07 uniform block capacity");
    bytes_.reserve(capacity);
  }
  bool CanAppend(size_t size,size_t alignment)const{
    if(!size||size%4||!alignment||alignment%4)throw std::invalid_argument("RW07 uniform alignment/size");
    if(sealed_)return false;
    const size_t padding=(alignment-bytes_.size()%alignment)%alignment;
    return padding<=capacity_-bytes_.size()&&size<=capacity_-bytes_.size()-padding;
  }
  size_t Append(const uint8_t* source,size_t size,size_t alignment){
    if(!source)throw std::invalid_argument("RW07 uniform bytes missing");
    if(!CanAppend(size,alignment))throw std::length_error("RW07 uniform block exhausted/sealed");
    const size_t padding=(alignment-bytes_.size()%alignment)%alignment;
    const size_t offset=bytes_.size()+padding;
    bytes_.resize(offset,0);
    bytes_.insert(bytes_.end(),source,source+size);
    return offset;
  }
  void Seal(){sealed_=true;}
  // The caller must hold the sole block reference after checked queue completion.
  // An encoded/submitted command may still refer to every previous offset.
  // QueueSubmit has consumed all earlier reads before a later QueueWriteBuffer.
  // Caller must additionally establish sole CPU ownership of this block.
  void ResetAfterOrderedSubmit(){ResetAfterCheckedCompletion();}
  void ResetAfterCheckedCompletion(){
    if(!sealed_)throw std::logic_error("RW07 unsubmitted uniform block recycle");
    bytes_.clear();
    sealed_=false;
  }
  bool Sealed()const{return sealed_;}
  const std::vector<uint8_t>& Bytes()const{return bytes_;}
 private:
  size_t capacity_;
  bool sealed_=false;
  std::vector<uint8_t> bytes_;
};
}  // namespace onnxruntime::webgpu
