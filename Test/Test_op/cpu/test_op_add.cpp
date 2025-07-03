#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"

TEST(test_add_cpu, add1_nostream) {
  int32_t size = 32 * 32;
  auto alloc_cpu = kuiper_base::CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1(kuiper_base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor t2(kuiper_base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out(kuiper_base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_EQ(t2.is_empty(), false);
  ASSERT_EQ(out.is_empty(), false);
  float* p1 = t1.ptr<float>();
  float* p2 = t2.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    *(p1 + i) = 1.f;
    *(p2 + i) = 1.f;
  }

  kernel::get_add_kernel(kuiper_base::DeviceType::kDeviceCPU)(t1, t2, out, nullptr);
  float* p3 = out.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(p3 + i), 2.f);
  }
}




