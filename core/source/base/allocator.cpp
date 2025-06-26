#include "base/allocator.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>

namespace kuiper_base{

/**
 * @brief CPU与Cuda的内存拷贝。
 */

void DeviceAllocator::memcpy(const void* src_ptr,void* dst_ptr,size_t byte_size,
                            MemcpyKind memcpy_kind,
                            void* stream,bool need_sync) const{
  // todo: stc与dst空指针检查
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dst_ptr, nullptr);

  if (!byte_size) {
    return;
  }
  // ?: 添加Ascend后端，是否需要在allocator派生类中实现copy,而不是在基类中实现
  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<CUstream_st*>(stream);
  }
  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
    std::memcpy(dst_ptr, src_ptr, byte_size);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    if (!stream_) {
      cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
    } else {
      cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    if (!stream_) {
      cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    if (!stream_) {
      cudaMemcpy(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
    }
  } else {
    LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
  }
  if (need_sync) {
    cudaDeviceSynchronize(); // cuda后端与主机同步
  }
}

/**
 * @brief 内存置零
 */
void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream,
                                  bool need_sync) {
  CHECK(device_type_ != DeviceType::kDeviceUnknown);
  if (device_type_ == DeviceType::kDeviceCPU) {
    std::memset(ptr, 0, byte_size);
  } else {
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      cudaMemsetAsync(ptr, 0, byte_size, stream_);
    } else {
      cudaMemset(ptr, 0, byte_size);
    }
    if (need_sync) {
      cudaDeviceSynchronize();
    }
  }
}

} // end namespace kuiper_base