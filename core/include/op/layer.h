// #ifndef KUIPER_INCLUDE_OP_LAYER_H_
// #define KUIPER_INCLUDE_OP_LAYER_H_

// #include <base/cuda_config.h>
// #include <string>
// #include <vector>
// #include "base/base.h"
// #include "tensor/tensor.h"

// namespace op{
// class Layer;
// // ? 语言与视觉layer是否区分
// enum class LayerType : uint8_t{
//   kLayerUnknown = 0,
//   kLayerLinear = 1,
//   kLayerEncode = 2,
//   kLayerEmbedding = 3,
//   kLayerRMSNorm = 4,
//   kLayerMatmul = 5,
//   kLayerRoPe = 6,
//   kLayerMHA = 7,
//   kLayerSoftmax = 8,
//   kLayerAdd = 9,
//   kLayerSwiGLU = 10,

//   kLayerConv = 11,
//   kLayerRelu = 12,
//   kLayerSilu = 13,
//   kLayerSigmoid = 14,
//   kLayerMaxpooling = 15,
//   kLayerAdaptiveAvgpool = 16,
// };

// class BaseLayer {
//  public:
//   explicit BaseLayer(kuiper_base::DeviceType device_type, LayerType layer_type, kuiper_base::DataType data_type,std::string layer_name = "");

//   kuiper_base::DataType data_type() const;

//   LayerType layer_type() const;

//   virtual kuiper_base::Status init() = 0;

//   virtual kuiper_base::Status forward() = 0;

//   virtual kuiper_base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

//   virtual kuiper_base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
//                                const tensor::Tensor& output1) = 0;

//   virtual kuiper_base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
//                                const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;

//   virtual kuiper_base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
//                                const tensor::Tensor& input3, const tensor::Tensor& input4,
//                                const tensor::Tensor& output1) = 0;

//   virtual kuiper_base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
//                                const tensor::Tensor& input3, const tensor::Tensor& input4,
//                                const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

//   virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

//   virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

//   virtual size_t input_size() const = 0;

//   virtual size_t output_size() const = 0;

//   virtual kuiper_base::Status check() const = 0;

//   virtual tensor::Tensor& get_input(int32_t idx) = 0;

//   virtual tensor::Tensor& get_output(int32_t idx) = 0;

//   virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

//   virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

//   virtual kuiper_base::Status set_weight(int32_t idx, const tensor::Tensor& weight);

//   virtual kuiper_base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
//                                   const void* weight_ptr,
//                                   kuiper_base::DeviceType device_type = kuiper_base::DeviceType::kDeviceUnknown);

//   const std::string& get_layer_name() const;

//   void set_layer_name(const std::string& layer_name);

//   kuiper_base::DeviceType device_type() const;

//   void set_device_type(kuiper_base::DeviceType device_type);

//  protected:
//   std::string layer_name_;
//   LayerType layer_type_ = LayerType::kLayerUnknown;
//   kuiper_base::DataType data_type_ = kuiper_base::DataType::kDataTypeUnknown;
//   kuiper_base::DeviceType device_type_ = kuiper_base::DeviceType::kDeviceUnknown;
// };
// }  // namespace op
// #endif  // KUIPER_INCLUDE_OP_LAYER_H_