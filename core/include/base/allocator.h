#ifndef KUIPER_INCLUDE_BASE_ALLOC_H_
#define KUIPER_INCLUDE_BASE_ALLOC_H_
#include "base/base.h"
#include <map>
#include <memory>
#include <vector> 

namespace kuiper_base {

/**
 * @brief 数据流向类
 */
enum class MemcpyKind : uint8_t {
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
  kMemcpyCPU2Ascend = 4,
  kMemcpyAscend2CPU = 5,
};

/**
 * @brief 后端设备的基类
 */
class DeviceAllocator {
 public:
   explicit DeviceAllocator(DeviceType type):device_type_(type){}
   virtual DeviceType device_type() const {return device_type_;}
   virtual void release(void* ptr) const = 0;
   virtual void* allocate(size_t byte_size) const = 0;

   virtual void memcpy(const void* src_ptr,void* dst_ptr,
            size_t byte_size,MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,void* stream = nullptr,bool need_sync = false) const;

   virtual void memset_zero(void* ptr,size_t byte_size,void* stream = nullptr,bool need_sync = false);

 private:
   DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

/**
 * @brief CPU后端
 */ 
class CPUDeviceAllocator : public DeviceAllocator{
  public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
};

/**
 * @brief CPU后端注册工厂，多CPU后端共享一个实例
 */ 
class CPUDeviceAllocatorFactory {
 public:
  static std::shared_ptr<CPUDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CPUDeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CPUDeviceAllocator> instance;
};

/**
 * @brief Cuda内存缓冲区，用于包装Cuda的内存指针，显示内存使用状态
 */
struct CudaMemoryBuffer {
  void* data;
  size_t byte_size;
  bool busy;

  CudaMemoryBuffer() = default;

  CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
      : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator{
  public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
  private:
    mutable std::map<int, size_t> no_busy_cnt_; // 未使用内存的计数
    mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_; // 大内存缓冲区
    mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_; // 小内存缓冲区
};

class CUDADeviceAllocatorFactory {
 public:
  static std::shared_ptr<CUDADeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<CUDADeviceAllocator>();
    }
    return instance;
  }

 private:
  static std::shared_ptr<CUDADeviceAllocator> instance;
};

// todo: 添加Ascend后端
class AscendDeviceAllocator : public DeviceAllocator{
    public:
    explicit AscendDeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
};

// todo: 添加Ascend后端工厂
class AscendDeviceAllocatorFactory {
 public:
  static std::shared_ptr<AscendDeviceAllocator> get_instance() {
    if (instance == nullptr) {
      instance = std::make_shared<AscendDeviceAllocator>();
    }
    return instance;
  }
 private:
  static std::shared_ptr<AscendDeviceAllocator> instance;
};

} // end namespace kuiper_base
#endif  // KUIPER_INCLUDE_BASE_ALLOC_H_

