#include <base/base.h>
#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/mha_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/rope_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/swiglu_kernel.h"
#include "cpu/scale_kernel.h"
#include "cpu/scale_sum_kernel.h"

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

EmbeddingKernel get_emb_kernel(kuiper_base::DeviceType device_type) {
  if (device_type == kuiper_base::DeviceType::kDeviceCPU) {
    return emb_kernel_normal;
  } else if (device_type == kuiper_base::DeviceType::kDeviceCUDA) {
    LOG(FATAL) << "Cuda device is not implemented.";
    return nullptr;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}

MatmulKernel get_matmul_kernel(kuiper_base::DeviceType device_type) {
  if (device_type == kuiper_base::DeviceType::kDeviceCPU) {
    return matmul_kernel_cpu;
  } else if (device_type == kuiper_base::DeviceType::kDeviceCUDA) {
    LOG(FATAL) << "Cuda device is not implemented.";
    return nullptr;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}

RMSNormKernel get_rmsnorm_kernel(kuiper_base::DeviceType device_type) {
  if (device_type == kuiper_base::DeviceType::kDeviceCPU) {
    return rmsnorm_kernel_cpu;
  } else if (device_type == kuiper_base::DeviceType::kDeviceCUDA) {
    LOG(FATAL) << "Cuda device is not implemented.";
    return nullptr;
  } else {
    LOG(FATAL) << "Unknown device type for get a add kernel.";
    return nullptr;
  }
}

SwigluKernel get_swiglu_kernel(kuiper_base::DeviceType device_type, void* stream) {
  if (device_type == kuiper_base::DeviceType::kDeviceCPU) {
    return swiglu_kernel_cpu;
  } else if (device_type == kuiper_base::DeviceType::kDeviceCUDA) {
    LOG(FATAL) << "Cuda device is not implemented.";
    return nullptr;
  } else {
    LOG(FATAL) << "Unknown device type for get a swiglu kernel.";
    return nullptr;
  }
}

SoftmaxInplaceKernel get_softmax_kernel(kuiper_base::DeviceType device_type) {
  if (device_type == kuiper_base::DeviceType::kDeviceCPU) {
    return softmax_inplace_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get an softmax kernel.";
    return nullptr;
  }
}

MHAKernel get_mha_kernel(kuiper_base::DeviceType device_type) {
  if (device_type == kuiper_base::DeviceType::kDeviceCPU) {
    return mha_kernel;
  } else if (device_type == kuiper_base::DeviceType::kDeviceCUDA) {
    LOG(FATAL) << "Cuda device is not implemented.";
    return nullptr;
  } else {
    LOG(FATAL) << "Unknown device type for get an mha kernel.";
    return nullptr;
  }
}

RoPEKernel get_rope_kernel(kuiper_base::DeviceType device_type) {
  if (device_type == kuiper_base::DeviceType::kDeviceCPU) {
    return rope_kernel_cpu;
  } else if (device_type == kuiper_base::DeviceType::kDeviceCUDA) {
    LOG(FATAL) << "Cuda device is not implemented.";
    return nullptr;
  } else {
    LOG(FATAL) << "Unknown device type for get a rope kernel.";
    return nullptr;
  }
}

ScaleKernel get_scale_kernel(kuiper_base::DeviceType device_type) {
  if (device_type == kuiper_base::DeviceType::kDeviceCPU) {
    return scale_inplace_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get a rope kernel.";
    return nullptr;
  }
}

ScaleSumKernel get_scale_sum_kernel(kuiper_base::DeviceType device_type) {
  if (device_type == kuiper_base::DeviceType::kDeviceCPU) {
    return scale_sum_kernel_cpu;
  } else {
    LOG(FATAL) << "Unknown device type for get a scale and reduce kernel.";
    return nullptr;
  }
}

}