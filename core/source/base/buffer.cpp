#include "base/buffer.h"
#include <glog/logging.h>

namespace kuiper_base{

Buffer::Buffer(size_t byte_size,std::shared_ptr<DeviceAllocator> allocator,
                void* ptr,bool use_external)
    : byte_size_(byte_size),allocator_(allocator),
    ptr_(ptr),use_external_(use_external){
    // 如果创建Buffer时，ptr为空，则说明是内部申请内存，内存生命周期由Buffer管理
    if(ptr == nullptr && allocator_){
        use_external_ = false;
        device_type_ = allocator_->device_type();
        ptr_ = allocator_->allocate(byte_size_);
    }
}

// 由各个后端对应的allocator释放内存
Buffer::~Buffer(){
    if (!use_external_) {
        if (ptr_ && allocator_) {
            allocator_->release(ptr_);
            ptr_ = nullptr;
        }
    }
}

void* Buffer::ptr(){
    return ptr_;
}

const void* Buffer::ptr() const{
    return ptr_;
}

size_t Buffer::byte_size() const{
    return byte_size_;
}

std::shared_ptr<DeviceAllocator> Buffer::allocator() const{
    return allocator_;
}

DeviceType Buffer::device_type() const{
    return device_type_;
}

void Buffer::set_device_type(DeviceType device_type){
    device_type_ = device_type;
}

std::shared_ptr<Buffer> Buffer::get_shared_from_this(){
    return shared_from_this();
}

bool Buffer::is_external() const{
    return this->use_external_;
}

bool Buffer::allocate(){
    if(allocator_ && byte_size_ != 0){
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size_);
        if(!ptr_){
            LOG(ERROR) << "Failed to allocate memory for buffer";
            return false;
        }
        return true;
    }else{
        LOG(ERROR) << "allocator is nullptr or byte_size_ is 0";
        return false;
    }
}

// TODO: 添加copy_from函数
void Buffer::copy_from(const Buffer* buffer) const {
  CHECK(allocator_ != nullptr);
  CHECK(buffer != nullptr || buffer->ptr() != nullptr);

  size_t dest_size = byte_size_;
  size_t src_size = buffer->byte_size();
  size_t byte_size = src_size < dest_size ? src_size : dest_size;

  const DeviceType& buffer_device = buffer->device_type();
  const DeviceType& current_device = this->device_type_;

  CHECK(buffer_device != DeviceType::kDeviceUnknown &&
        current_device != DeviceType::kDeviceUnknown);
  
  if (buffer_device == DeviceType::kDeviceCPU &&
      current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size);
  } else if (buffer_device == DeviceType::kDeviceCUDA &&
             current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCUDA2CPU);
  } else if (buffer_device == DeviceType::kDeviceCPU &&
             current_device == DeviceType::kDeviceCUDA) {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCPU2CUDA);
  } else {
    return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCUDA2CUDA);
  }
}

void Buffer::copy_from(const Buffer& buffer) const {
  CHECK(allocator_ != nullptr);
  CHECK(buffer.ptr_ != nullptr);

  size_t byte_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
  const DeviceType& buffer_device = buffer.device_type();
  const DeviceType& current_device = this->device_type();
  CHECK(buffer_device != DeviceType::kDeviceUnknown &&
        current_device != DeviceType::kDeviceUnknown);

  if (buffer_device == DeviceType::kDeviceCPU &&
      current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size);
  } else if (buffer_device == DeviceType::kDeviceCUDA &&
             current_device == DeviceType::kDeviceCPU) {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCUDA2CPU);
  } else if (buffer_device == DeviceType::kDeviceCPU &&
             current_device == DeviceType::kDeviceCUDA) {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCPU2CUDA);
  } else {
    return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                              MemcpyKind::kMemcpyCUDA2CUDA);
  }
}
} // end namespace kuiper_base  