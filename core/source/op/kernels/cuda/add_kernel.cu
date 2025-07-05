#include "add_kernel.cuh"
#define FLOAT4(x) reinterpret_cast<const float4*>(&(x))[0]

namespace kernel{   
void __global__ add_kernel_cu(const float* input1, const float* input2, 
                        float* output, int size){
  int index = 4*(blockIdx.x * blockDim.x + threadIdx.x);
  if(index >= size){
    return;
  }
  float4 reg_a = FLOAT4(input1[index]);
  float4 reg_b = FLOAT4(input2[index]);
  float4& reg_c = (float4&)(output[index]);
  
  reg_c.x = reg_a.x + reg_b.x;
  reg_c.y = reg_a.y + reg_b.y;
  reg_c.z = reg_a.z + reg_b.z;
  reg_c.w = reg_a.w + reg_b.w;
  
  (float4&)(output[index]) = reg_c;
}

void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                   const tensor::Tensor& output, void* stream){
  CHECK_EQ(input1.is_empty(), false);
  CHECK_EQ(input2.is_empty(), false);
  CHECK_EQ(output.is_empty(), false);

  int32_t size = static_cast<int32_t>(input1.size());
  CHECK_EQ(size, input2.size());
  CHECK_EQ(size, output.size());
  
  const int threads = 512;
  const int blocks = (size + threads*4 - 1) / (threads*4);
  if(stream){
    cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
    add_kernel_cu<<<blocks, threads, 0, stream_>>>(input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()), size);
  }else{
    add_kernel_cu<<<blocks, threads>>>(input1.ptr<float>(), input2.ptr<float>(), const_cast<float*>(output.ptr<float>()), size);
  }
}

}//namespace kernel
