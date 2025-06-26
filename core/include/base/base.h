#ifndef KUIPER_INCLUDE_BASE_BASE_H_
#define KUIPER_INCLUDE_BASE_BASE_H_
#include <glog/logging.h>
#include <cstdint>
#include <string>

namespace model {
/**
 * @brief 定义模型数据缓冲区的类型
 * 
 * 该枚举列出了模型在推理过程中使用的不同类型的缓冲区，
 * 如输入token、KV cache等
 */
enum class ModelBufferType{

};

}// end namespace model

// 将您的命名空间定义为kuiper_base，避免与glog的base命名空间冲突
namespace kuiper_base{
/**
 * @brief 支持后端设备类型(CPU、Cuda、Ascend)
 */
enum class DeviceType : uint8_t {
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
  kDeviceAscend = 3
};

/**
 * @brief 支持数据类型
 */
enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeInt8 = 2,
  kDataTypeInt32 = 3,
};

/**
 * @brief 支持模型类型
 */
enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,
  kModelTypeLLama2 = 1,
};

/**
 * @brief 代码状态类型
 */
enum StatusCode : uint8_t {
  kSuccess = 0,
  kFunctionUnImplement = 1,
  kPathNotValid = 2,
  kModelParseError = 3,
  kInternalError = 5,
  kKeyValueHasExist = 6,
  kInvalidArgument = 7,
};

/**
 * @brief 获取数据类型的byte数量
 */
inline size_t DataTypeSize(DataType data_type) {
  if (data_type == DataType::kDataTypeFp32) {
    return sizeof(float);
  } else if (data_type == DataType::kDataTypeInt8) {
    return sizeof(int8_t);
  } else if (data_type == DataType::kDataTypeInt32) {
    return sizeof(int32_t);
  } else {
    return 0;
  }
}

/*
 * @brief 禁止拷贝的基类,禁止通过拷贝构造函数/赋值操作来创建新对象
*/
class NoCopyable {
 protected:
  NoCopyable() = default;

  ~NoCopyable() = default;

  NoCopyable(const NoCopyable&) = delete;

  NoCopyable& operator=(const NoCopyable&) = delete;
};

} // end namespace kuiper_base

// 创建一个别名让base指向kuiper_base命名空间
namespace base = kuiper_base;

#endif  // KUIPER_INCLUDE_BASE_BASE_H_


