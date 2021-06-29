#ifndef EFFICIENTDET_H
#define EFFICIENTDET_H

#include "torch/torch.h"

#include "efficientdettype.h"
#include "efficientdetconfig.h"

namespace dldetection {

class EfficientDetImpl : public torch::nn::Module
{
public:
    EfficientDetImpl(int numClass = 80, EfficientDetType type = EfficientDetType::EfficientDetB0, bool loadWeights = false);

    /*!
     * \brief 前向
     * \param x 输入
     * \return
     */
    torch::Tensor forward(torch::Tensor x);

    /*!
     * \brief FreezeBN 冻结BN层
     */
    void FreezeBN();

private:
    int compoundCoef;

};
TORCH_MODULE(EfficientDet);
}

#endif // EFFICIENTDET_H
