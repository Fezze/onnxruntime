// Source-private RW07 schema2. Host helpers contain opaque administration only;
// logical dimensions, count, status and premises are produced on GPU.
#pragma once
#include <array>
#include <cstdint>

namespace onnxruntime::webgpu::m0_extent_abi2 {
constexpr uint32_t Schema = 2;
constexpr uint32_t Words = 32;
constexpr uint32_t Bytes = Words * sizeof(uint32_t);
constexpr uint32_t MaxRank = 8;
constexpr uint32_t SchemaWord = 0;
constexpr uint32_t RankWord = 1;
constexpr uint32_t DimensionsWord = 2;
constexpr uint32_t ElementsWord = 18;
constexpr uint32_t StatusWord = 20;
constexpr uint32_t PremisesWord = 21;
constexpr uint32_t SessionWord = 22;
constexpr uint32_t RunWord = 24;
constexpr uint32_t SidecarGenerationWord = 26;
constexpr uint32_t CarrierGenerationWord = 28;
constexpr uint32_t ProducerRoleWord = 30;
constexpr uint32_t ProducerInstanceWord = 31;
constexpr uint32_t DurationRole = 1, FrameRole = 2, ReconstructionRole = 3;
constexpr uint32_t DurationInstance = 1001, FrameInstance = 1002, ReconstructionInstance = 1003;
constexpr bool CertifiedProducer(uint32_t role, uint32_t instance) {
  return (role == DurationRole && instance == DurationInstance) ||
         (role == FrameRole && instance == FrameInstance) ||
         (role == ReconstructionRole && instance == ReconstructionInstance);
}

// These ten words are shader uniforms for comparison/stamping, NOT a device
// extent readback or a host-built logical sidecar. The caller obtains current
// Session/Run from the private core bridge and generations from actual live
// bound resource descriptors. Role/instance come from the certified edge.
struct OpaqueOwnership {
  uint64_t session = 0, run = 0, sidecar_generation = 0, carrier_generation = 0;
  uint32_t producer_role = 0, producer_instance = 0;
  constexpr bool Valid() const {
    return session && run && sidecar_generation && carrier_generation &&
           sidecar_generation != carrier_generation && CertifiedProducer(producer_role, producer_instance);
  }
  constexpr std::array<uint32_t, 10> UniformWords() const {
    return {uint32_t(session), uint32_t(session >> 32), uint32_t(run), uint32_t(run >> 32),
            uint32_t(sidecar_generation), uint32_t(sidecar_generation >> 32),
            uint32_t(carrier_generation), uint32_t(carrier_generation >> 32),
            producer_role, producer_instance};
  }
};
static_assert(ProducerInstanceWord + 1 == Words && Bytes == 128);
}  // namespace onnxruntime::webgpu::m0_extent_abi2
