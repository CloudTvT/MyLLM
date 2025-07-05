#include <base/base.h>
#include "cpu/add_kernel.h"
#include "cuda/add_kernel.cuh"
#include "kernels_interface.h"

namespace kernel{
AddKernel get_add_kernel(kuiper_base::DeviceType device_type) {
  if (device_type == kuiper_base::DeviceType::kDeviceCPU) {
    return add_kernel_cpu;
  } else if (device_type == kuiper_base::DeviceType::kDeviceCUDA) {
    return add_kernel_cu;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}
}