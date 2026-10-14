// rw07-fullgpu-v1 canonical device sidecar ABI. Source-private, version 1.
#pragma once
#include <cstdint>
namespace rw07 {
constexpr uint32_t ExtentSchema = 1;
constexpr uint32_t MaxRank = 8;
// Device-only UINT32[24]: schema, rank, eight paired-u32 INT64 dimensions,
// paired-u32 logical element count, status, validated premises, owner epoch.
constexpr uint32_t ExtentWords = 24;
constexpr uint32_t LogicalDimensionsWord = 2;
constexpr uint32_t LogicalElementsWord = 18;
constexpr uint32_t StatusWord = 20;
constexpr uint32_t PremisesWord = 21;
constexpr uint32_t EpochWord = 22;
enum Premise : uint32_t {
  SigmoidFiniteRange = 1u << 0,
  ReductionFiniteRange = 1u << 1,
  SpeedSupported = 1u << 2,
  DurationIntegralRange = 1u << 3,
  PrefixChecked = 1u << 4,
  WithinCapacity = 1u << 5,
};
constexpr uint32_t DurationPremises = SigmoidFiniteRange | ReductionFiniteRange |
    SpeedSupported | DurationIntegralRange | PrefixChecked | WithinCapacity;
// Administrative status classification must not turn an unsupported resource or
// numerical premise into a claimed original CPU semantic rejection.
enum class ExtentStatus : uint32_t {
  Valid = 0,
  OriginalEmptyGather = 1,
  NonfiniteOrRangePremise = 100,
  UnsupportedSpeed = 101,
  ResourceCapacity = 102,
  InvalidEpochOrSchema = 103,
};
} // namespace rw07
