#ifndef KUIPER_INCLUDE_BASE_BUFFER_H_
#define KUIPER_INCLUDE_BASE_BUFFER_H_

#include <memory>
#include "base/allocator.h"

namespace kuiper_base{
/* 
    @brief 缓冲区基类,用于对allocator申请的后端内存进行管理
*/
class Buffer : public NoCopyable,std::enable_shared_from_this<Buffer>{
  private:
    size_t byte_size_ = 0;
    void* ptr_ = nullptr;
    bool use_external_ = false;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> allocator_ = nullptr;

  public:
    explicit Buffer() = default;
    explicit Buffer(size_t byte_size,std::shared_ptr<DeviceAllocator> allocator,
                    void* ptr = nullptr,bool use_external = false);
    virtual ~Buffer(); 

    bool allocate();

    void copy_from(const Buffer& src_buffer) const;
    void copy_from(const Buffer* src_buffer) const; // 显式传递地址，允许nullptr

    void* ptr();

    const void* ptr() const; // 常量对象调用

    size_t byte_size() const;

    std::shared_ptr<DeviceAllocator> allocator() const;

    DeviceType device_type() const;

    void set_device_type(DeviceType device_type); // 设置设备类型

    std::shared_ptr<Buffer> get_shared_from_this(); // 获取共享指针

    bool is_external() const;

};
} // end namespace kuiper_base

#endif  // KUIPER_INCLUDE_BASE_BUFFER_H_    