// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Authored for the final consolidated batch; no validation during implementation.
#include <cmath>
#include <limits>
#include <vector>
#include "gtest/gtest.h"
#include "default_providers.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime::test {
TEST(Pow_WebGPU, OriginalAlbertCubeUsesTwoRoundedMultiplications) {
  auto ep = DefaultWebGpuExecutionProvider();
  if (!ep) GTEST_SKIP() << "WebGPU execution provider is unavailable";
  const std::vector<float> input{-7.125f, -2.31f, -0.004f, -0.0f, 0.0f, 0.004f, 0.71f, 1.7f, 13.25f};
  std::vector<float> expected;
  expected.reserve(input.size());
  for (const float value : input) {
    const float square = value * value;
    expected.push_back(square * value);
  }
  OpTester test("Pow", 15);
  test.AddInput<float>("X", {int64_t(input.size())}, input);
  test.AddInput<float>("exponent", {}, {3.0f});
  test.AddOutput<float>("Y", {int64_t(expected.size())}, expected);
  test.SetOutputAbsErr("Y", 0.0f);
  test.SetOutputRelErr("Y", 0.0f);
  test.SetCustomOutputVerifier([expected](const std::vector<OrtValue>& fetched, const std::string&) {
    ASSERT_EQ(fetched.size(), 1u);
    const auto* actual = fetched[0].Get<Tensor>().Data<float>();
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(actual[i], expected[i]) << "element " << i;
    EXPECT_TRUE(std::signbit(actual[3]));
    EXPECT_FALSE(std::signbit(actual[4]));
  });
  test.ConfigEp(std::move(ep)).RunWithConfig();
}
TEST(Pow_WebGPU, CubePreservesNonfiniteAndSignedZeroClasses) {
  auto ep = DefaultWebGpuExecutionProvider();
  if (!ep) GTEST_SKIP() << "WebGPU execution provider is unavailable";
  const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
  OpTester test("Pow", 15);
  test.AddInput<float>("X", {5}, {-inf, inf, nan, -0.0f, 0.0f});
  test.AddInput<float>("exponent", {}, {3.0f});
  test.AddOutput<float>("Y", {5}, {-inf, inf, nan, -0.0f, 0.0f});
  test.SetCustomOutputVerifier([](const std::vector<OrtValue>& fetched, const std::string&) {
    ASSERT_EQ(fetched.size(), 1u);
    const auto* actual = fetched[0].Get<Tensor>().Data<float>();
    EXPECT_TRUE(std::isinf(actual[0]) && std::signbit(actual[0]));
    EXPECT_TRUE(std::isinf(actual[1]) && !std::signbit(actual[1]));
    EXPECT_TRUE(std::isnan(actual[2]));
    EXPECT_EQ(actual[3], 0.0f);
    EXPECT_TRUE(std::signbit(actual[3]));
    EXPECT_EQ(actual[4], 0.0f);
    EXPECT_FALSE(std::signbit(actual[4]));
  });
  test.ConfigEp(std::move(ep)).RunWithConfig();
}
}  // namespace onnxruntime::test
