#include "efficientdet.h"
#include "utils.h"

namespace dldetection {

EfficientDetImpl::EfficientDetImpl(int numClass, EfficientDetType type, bool loadWeights)
{

}

/*!
 * \brief 前向
 * \param x 输入
 * \return
 */
at::Tensor EfficientDetImpl::forward(at::Tensor x)
{

}

/*!
 * \brief FreezeBN 冻结BN层
 */
void EfficientDetImpl::FreezeBN()
{
    for (size_t i = 0; i < this->modules().size(); ++i)
    {
        if(InstanceOf<torch::nn::BatchNorm2d>(&this->modules()[i]))
        {
            this->modules()[i]->eval();
        }
    }
}

}
