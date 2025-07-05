#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../source/op/kernels/kernels_interface.h"
#include "base/buffer.h"
#include "../utils.cuh"

TEST(test_op_add, add_cpu_nostream) {
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

TEST(test_op_add, add_cuda_nostream) {
  int32_t size = 32 * 32;
  auto alloc_cuda = kuiper_base::CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1(kuiper_base::DataType::kDataTypeFp32, size, true, alloc_cuda);
  tensor::Tensor t2(kuiper_base::DataType::kDataTypeFp32, size, true, alloc_cuda);
  tensor::Tensor out(kuiper_base::DataType::kDataTypeFp32, size, true, alloc_cuda);

  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_EQ(t2.is_empty(), false);
  ASSERT_EQ(out.is_empty(), false);

  
  float* p1 = t1.ptr<float>();
  float* p2 = t2.ptr<float>();
  set_value_cu(p1,size,1.f);
  set_value_cu(p2,size,1.f);

  kernel::get_add_kernel(kuiper_base::DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
  cudaDeviceSynchronize();// 主机与GPU同步
  float* p3 = new float[size];
  cudaMemcpy(p3,out.ptr<float>(),sizeof(float)*size,cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(p3 + i), 2.f);
  }

  delete[] p3;
}




