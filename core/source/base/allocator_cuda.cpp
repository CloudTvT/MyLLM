#include <cuda_runtime_api.h>
#include "base/allocator.h"

namespace kuiper_base{
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {}

// TODO: 添加cuda后端内存申请
void* CUDADeviceAllocator::allocate(size_t byte_size) const {
  int id = -1;
  cudaError state = cudaGetDevice(&id);
  CHECK(state == cudaSuccess) << "Failed to get current device";
  if(byte_size >= 1024*1024){
    // 大内存申请，先在显存池中寻找合适的内存
    auto& big_buffers = big_buffers_map_[id];
    int selec_index = -1;
    for(int i = 0;i<big_buffers.size();i++){
      if(big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy && 
      big_buffers[i].byte_size - byte_size < 1*1024*1024){
        if(selec_index == -1 || big_buffers[i].byte_size < big_buffers[selec_index].byte_size){
          selec_index = i;
        }
      }
    }
    if(selec_index != -1){
      big_buffers[selec_index].busy = true;
      return big_buffers[selec_index].data;
    }
    // 如果显存池中没有合适的内存，则申请新的内存
    void* ptr = nullptr;
    state = cudaMalloc(&ptr,byte_size);
    if(state != cudaSuccess){
      char buf[256];
      snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
               "left on  device.",
               byte_size >> 20);
      LOG(ERROR) << buf;
      return nullptr;
    }
    big_buffers.emplace_back(CudaMemoryBuffer(ptr,byte_size,true));
    return ptr;
  }
  // 小内存申请，先在显存池中寻找合适的内存
  auto& cuda_buffers = cuda_buffers_map_[id];
  for(int i = 0;i<cuda_buffers.size();i++){
    if(cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy){
      cuda_buffers[i].busy = true;
      // 更新未使用内存的大小
      no_busy_cnt_[id] -= cuda_buffers[i].byte_size;  
      return cuda_buffers[i].data;
    }
  }
  // 如果显存池中没有合适的内存，则申请新的内存
  void* ptr = nullptr;
  state = cudaMalloc(&ptr,byte_size);
  if(state != cudaSuccess){
    char buf[256];
    snprintf(buf, 256,
               "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
               "left on  device.",
               byte_size >> 20);
    LOG(ERROR) << buf;
    return nullptr;
  }
  cuda_buffers.emplace_back(CudaMemoryBuffer(ptr,byte_size,true));
  return ptr;
}

// TODO: 添加cuda后端内存释放
void CUDADeviceAllocator::release(void* ptr) const {
  if (ptr){
    return;
  }
  if(cuda_buffers_map_.empty()){
    return;
  }
  cudaError_t state = cudaSuccess;
  for(auto& it:cuda_buffers_map_){
    if(no_busy_cnt_[it.first] > 1024*1024*1024){
      auto& cuda_buffers = it.second;
      std::vector<CudaMemoryBuffer> temp;
      for(int i = 0;i<cuda_buffers.size();i++){
        if(!cuda_buffers[i].busy){
          state = cudaSetDevice(it.first);
          state = cudaFree(cuda_buffers[i].data);
        } else {
          temp.push_back(cuda_buffers[i]);
        } 
      }
      cuda_buffers.clear();
      it.second = temp;
      no_busy_cnt_[it.first] = 0;
    }
  }
  
  for(auto& it : cuda_buffers_map_){
    auto& cuda_buffers = it.second;
    for(int i = 0;i < cuda_buffers.size();i++){
      if(cuda_buffers[i].data == ptr){
        no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
        cuda_buffers[i].busy = false;
        return;
      }
    }

    auto& big_buffers = big_buffers_map_[it.first];
    for(int i = 0;i < big_buffers.size();i++){
      if(big_buffers[i].data == ptr){
        big_buffers[i].busy = false;
        return;
      }
    }
  }
  state = cudaFree(ptr);
  CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

} // end namespace kuiper_base  