// #include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"

TEST(test_buffer, allocate) {
  using namespace kuiper_base;
  auto alloc = CPUDeviceAllocatorFactory::get_instance();
  Buffer buffer(32, alloc);
  ASSERT_NE(buffer.ptr(), nullptr);
}

TEST(test_buffer, allocate_cuda) {
  using namespace kuiper_base;
  auto alloc = CUDADeviceAllocatorFactory::get_instance();
  Buffer buffer(32, alloc);
  ASSERT_NE(buffer.ptr(), nullptr);
}

TEST(test_buffer, use_external) {
  using namespace kuiper_base;
  auto alloc = CPUDeviceAllocatorFactory::get_instance();
  float* ptr = new float[32];
  Buffer buffer(32, nullptr, ptr, true);
  ASSERT_EQ(buffer.is_external(), true);
  delete[] ptr;
}

