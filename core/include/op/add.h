#ifndef KUIPER_INCLUDE_OP_ADD_H
#define KUIPER_INCLUDE_OP_ADD_H
#include "base/base.h"
#include "layer.h"
namespace op{

class VecAddLayer : public Layer{
  public:
    explicit VecAddLayer(kuiper_base::DeviceType device_type);
    kuiper_base::Status forward() override;
    kuiper_base::Status check() const override;
};
}
#endif