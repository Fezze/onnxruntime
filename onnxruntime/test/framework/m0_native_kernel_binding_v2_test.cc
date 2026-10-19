// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/plugin_ep/m0_native_kernel_inspection_v2.h"
#include "core/session/plugin_ep/ep_kernel_registration.h"
#include "asserts.h"
#include "gtest/gtest.h"

namespace onnxruntime::test {
namespace {
struct CreatorCounts { size_t entered = 0; };
OrtStatus* ORT_API_CALL NeverCreate(void* state, const OrtKernelInfo*, OrtKernelImpl**) noexcept {
  ++static_cast<CreatorCounts*>(state)->entered;
  return nullptr;
}
std::unique_ptr<KernelDef> AddDefinition(OrtMemType output_memory = OrtMemTypeDefault) {
  KernelDefBuilder builder;
  builder.SetName("Add").SetDomain("").SinceVersion(14)
      .Provider(kWebGpuExecutionProvider).TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  builder.OutputMemoryType(output_memory, 0);
  return builder.Build();
}
struct RegisteredFixture {
  CreatorCounts calls;
  std::unique_ptr<KernelDef> original = AddDefinition();
  KernelRegistry provider, selected;
  M0NativeProviderKernelBindingV2 expected;
  RegisteredFixture() {
    ORT_THROW_IF_ERROR(provider.Register(MakePluginEpKernelCreateInfo(original.get(), NeverCreate, &calls)));
    ORT_THROW_IF_ERROR(selected.Register(MakePluginEpKernelCreateInfo(original.get(), NeverCreate, &calls)));
    expected.abi = 2;
    expected.record_id = 1;
    expected.type = M0NativeKernelTypeV2::Float32;
    expected.creator = {NeverCreate, &calls};
    expected.registry = reinterpret_cast<const OrtKernelRegistry*>(&provider);
    expected.registration_definition = reinterpret_cast<const OrtKernelDef*>(original.get());
  }
  const KernelCreateInfo& Actual() const { return selected.GetKernelCreateMap().begin()->second; }
};
void ExpectUnchangedRefusal(const RegisteredFixture& fixture, const KernelRegistry& registry,
                            const KernelCreateInfo& actual, const M0NativeProviderKernelBindingV2& expected) {
  M0NativeSelectedKernelBindingV2 output;
  output.provider.record_id = 999;
  output.selected_registry = &fixture.selected;
  EXPECT_FALSE(M0InspectNativeSelectedKernelV2(registry, actual, expected, output));
  EXPECT_EQ(output.provider.record_id, 999u);
  EXPECT_EQ(output.selected_registry, &fixture.selected);
  EXPECT_EQ(output.selected_create_info, nullptr);
  EXPECT_EQ(fixture.calls.entered, 0u);
}
}  // namespace

TEST(M0NativeKernelBindingV2, ActualRegisteredAndSelectedClonesKeepSeparateIdentitiesWithoutCallingCreator) {
  RegisteredFixture fixture;
  M0NativeSelectedKernelBindingV2 output;
  ASSERT_TRUE(M0InspectNativeSelectedKernelV2(fixture.selected, fixture.Actual(), fixture.expected, output));
  EXPECT_EQ(output.provider_registry, &fixture.provider);
  EXPECT_EQ(output.selected_registry, &fixture.selected);
  EXPECT_EQ(output.selected_create_info, &fixture.Actual());
  EXPECT_EQ(output.registered_definition, fixture.provider.GetKernelCreateMap().begin()->second.kernel_def.get());
  EXPECT_EQ(output.selected_definition, fixture.Actual().kernel_def.get());
  EXPECT_NE(output.registered_definition, output.selected_definition);
  EXPECT_NE(output.registered_definition, fixture.original.get());
  EXPECT_EQ(output.provider.creator.callback, NeverCreate);
  EXPECT_EQ(output.provider.creator.state, &fixture.calls);
  EXPECT_EQ(fixture.calls.entered, 0u);
}

TEST(M0NativeKernelBindingV2, RefusesBadAbiUnknownIdsTypesNullAndCallbackStateSubstitution) {
  RegisteredFixture fixture;
  for (int fault = 0; fault < 8; ++fault) {
    auto expected = fixture.expected;
    CreatorCounts foreign_state;
    if (fault == 0) expected.abi = 1;
    if (fault == 1) expected.record_id = 0;
    if (fault == 2) expected.record_id = 43;
    if (fault == 3) expected.type = M0NativeKernelTypeV2::Float32LstmInt32;
    if (fault == 4) expected.creator.callback = nullptr;
    if (fault == 5) expected.creator.state = &foreign_state;
    if (fault == 6) expected.registry = nullptr;
    if (fault == 7) expected.registration_definition = nullptr;
    ExpectUnchangedRefusal(fixture, fixture.selected, fixture.Actual(), expected);
    EXPECT_EQ(foreign_state.entered, 0u);
  }
}

TEST(M0NativeKernelBindingV2, RefusesSameNamedUnregisteredAndForeignFunctorDefinitions) {
  RegisteredFixture fixture;
  auto not_registered = MakePluginEpKernelCreateInfo(fixture.original.get(), NeverCreate, &fixture.calls);
  ExpectUnchangedRefusal(fixture, fixture.selected, not_registered, fixture.expected);
  KernelRegistry foreign;
  KernelCreateInfo fake(AddDefinition(), [&fixture](FuncManager&, const OpKernelInfo&, std::unique_ptr<OpKernel>&) {
    ++fixture.calls.entered;
    return Status::OK();
  });
  ASSERT_STATUS_OK(foreign.Register(std::move(fake)));
  ExpectUnchangedRefusal(fixture, foreign, foreign.GetKernelCreateMap().begin()->second, fixture.expected);
}

TEST(M0NativeKernelBindingV2, RefusesMemoryContractDriftAndAbsentProviderRegistration) {
  RegisteredFixture fixture;
  KernelRegistry changed;
  auto changed_definition = AddDefinition(OrtMemTypeCPUOutput);
  ASSERT_STATUS_OK(changed.Register(MakePluginEpKernelCreateInfo(changed_definition.get(), NeverCreate, &fixture.calls)));
  ExpectUnchangedRefusal(fixture, changed, changed.GetKernelCreateMap().begin()->second, fixture.expected);
  KernelRegistry empty;
  auto expected = fixture.expected;
  expected.registry = reinterpret_cast<const OrtKernelRegistry*>(&empty);
  ExpectUnchangedRefusal(fixture, fixture.selected, fixture.Actual(), expected);
}

TEST(M0NativeKernelBindingV2, ClosedRecordsExcludeFaultKernelsOtherDomainsAndUnboundedCustomVersions) {
  EXPECT_EQ(M0NativeKernelRecordByDefinitionV2("PredictorResidualFaultExtentV1", kM0NativeDomainV2, 1, 1), nullptr);
  EXPECT_EQ(M0NativeKernelRecordByDefinitionV2("Conv", "com.ms.internal.nhwc", 11, 21), nullptr);
  EXPECT_EQ(M0NativeKernelRecordByDefinitionV2("DecoderInputExtentV1", kM0NativeDomainV2, 1,
                                               kM0NativeOpenVersionV2), nullptr);
  EXPECT_EQ(M0NativeKernelRecordByIdV2(0), nullptr);
  EXPECT_EQ(M0NativeKernelRecordByIdV2(43), nullptr);
}
}  // namespace onnxruntime::test
