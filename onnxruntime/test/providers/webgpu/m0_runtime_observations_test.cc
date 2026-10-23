// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(ORT_M0_DESCRIPTOR_QUALIFICATION) && !defined(ORT_MINIMAL_BUILD)
#include <limits>
#include "gtest/gtest.h"
#include "core/common/m0_memory_reserve_v1.h"
#include "core/providers/webgpu/m0_fullgpu_resource_ownership.h"

namespace onnxruntime::test {
namespace {
using webgpu::M0FullGpuResourceOwnership;
using webgpu::M0WorkloadEventV1;

TEST(M0RuntimeObservations, AllocationChurnIsSeparateFromLiveAndPeak) {
  M0FullGpuResourceOwnership owner;
  const auto first=owner.Acquire(1,64);
  EXPECT_TRUE(owner.Retire(1,first));
  const auto second=owner.Acquire(1,32);
  EXPECT_FALSE(owner.Retire(1,first));
  auto s=owner.Statistics();
  EXPECT_EQ(s.live_requested_bytes,32u);
  EXPECT_EQ(s.peak_live_requested_bytes,64u);
  EXPECT_EQ(s.total_requested_bytes,96u);
  EXPECT_EQ(s.total_retired_bytes,64u);
  EXPECT_TRUE(owner.Retire(1,second));
  s=owner.Statistics();
  EXPECT_EQ(s.total_requested_bytes,s.total_retired_bytes);
  EXPECT_EQ(s.live_requested_bytes,0u);
}

TEST(M0RuntimeObservations, TransferPayloadAndCopyPaddingAreNotDoubleCounted) {
  M0FullGpuResourceOwnership owner;
  owner.RecordWorkload(M0WorkloadEventV1::Upload,101,104);
  owner.RecordWorkload(M0WorkloadEventV1::Upload,13,0);  // Direct mapped write, no GPU staging copy.
  owner.RecordWorkload(M0WorkloadEventV1::Download,19,32);
  owner.RecordWorkload(M0WorkloadEventV1::DeviceCopy,48);
  const auto s=owner.Statistics();
  EXPECT_EQ(s.upload_payload_bytes,114u);
  EXPECT_EQ(s.upload_copy_bytes,104u);
  EXPECT_EQ(s.download_payload_bytes,19u);
  EXPECT_EQ(s.download_copy_bytes,32u);
  EXPECT_EQ(s.device_copy_bytes,48u);
  EXPECT_EQ(s.total_requested_bytes,0u);
}

TEST(M0RuntimeObservations, OperatorsAndEncodedDispatchesHaveSeparateUnits) {
  M0FullGpuResourceOwnership owner;
  owner.RecordWorkload(M0WorkloadEventV1::OperatorStarted);
  owner.RecordWorkload(M0WorkloadEventV1::OperatorStarted);
  owner.RecordWorkload(M0WorkloadEventV1::OperatorCompleted);
  owner.RecordWorkload(M0WorkloadEventV1::Dispatch,12);
  owner.RecordWorkload(M0WorkloadEventV1::IndirectDispatch,999);  // Unknown indirect grid is excluded.
  const auto s=owner.Statistics();
  EXPECT_EQ(s.operators_started,2u);
  EXPECT_EQ(s.operators_completed,1u);
  EXPECT_EQ(s.dispatches_encoded,2u);
  EXPECT_EQ(s.workgroups_encoded,12u);
  EXPECT_EQ(s.indirect_dispatches_encoded,1u);
}

TEST(M0RuntimeObservations, CumulativeOverflowDoesNotStopExecutionOrWrap) {
  M0FullGpuResourceOwnership owner;
  const auto maximum=std::numeric_limits<uint64_t>::max();
  const auto first=owner.Acquire(1,maximum);
  ASSERT_TRUE(owner.Retire(1,first));
  const auto second=owner.Acquire(2,1);
  owner.RecordWorkload(M0WorkloadEventV1::Upload,maximum,maximum);
  owner.RecordWorkload(M0WorkloadEventV1::Upload,1,1);
  const auto s=owner.Statistics();
  EXPECT_EQ(s.total_requested_bytes,maximum);
  EXPECT_EQ(s.upload_payload_bytes,maximum);
  EXPECT_TRUE(s.counters_saturated);
  EXPECT_EQ(s.live_requested_bytes,1u);
  EXPECT_TRUE(owner.Retire(2,second));
}

TEST(M0RuntimeObservations, PlanningContinuesWhenOnlyQueuedSlackDoesNotFit) {
  constexpr uint64_t available=1383632896, future=649452931, minimum=118518816;
  const auto strict=M0MemoryPlanningBudgetV1(available,future,minimum,true);
  const auto monitor=M0MemoryPlanningBudgetV1(available,future,minimum,false);
  EXPECT_LT(strict,minimum);
  EXPECT_GE(monitor,minimum);
  EXPECT_EQ(monitor,available-kM0MemoryReserveBytesV1-future);
}

TEST(M0RuntimeObservations, MissingHeadroomIsWarningNotArtificialCapacityRefusal) {
  constexpr auto maximum=std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(M0MemoryPlanningBudgetV1(0,100,10,false),maximum);
  EXPECT_EQ(M0MemoryPlanningBudgetV1(0,100,10,true),0u);
  EXPECT_FALSE(M0MemoryHeadroomV1(0,maximum));
  EXPECT_TRUE(M0MemoryHeadroomV1(kM0MemoryReserveBytesV1+16,16));
  EXPECT_FALSE(M0MemoryHeadroomV1(kM0MemoryReserveBytesV1+16,17));
}

TEST(M0RuntimeObservations, LowRamRelaxesWarningReserveBeforeUsingHardwareOnlyCapacity) {
  constexpr uint64_t future=650*1024*1024, minimum=119*1024*1024;
  constexpr uint64_t available=future+300*1024*1024;
  EXPECT_EQ(M0MemoryPlanningBudgetV1(available,future,minimum,false),150u*1024*1024);
  EXPECT_LT(M0MemoryPlanningBudgetV1(available,future,minimum,true),minimum);
  EXPECT_EQ(M0MemoryPlanningBudgetV1(future+minimum,future,minimum,false),minimum);
}
}  // namespace
}  // namespace onnxruntime::test
#endif
