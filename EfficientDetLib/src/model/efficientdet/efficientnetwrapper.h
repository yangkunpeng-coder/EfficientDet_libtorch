#ifndef EFFICIENTNETWRAPPER_H
#define EFFICIENTNETWRAPPER_H

#include "torch/torch.h"

#include "efficientnet/efficientnet.h"
#include "efficientnet/efficientnettype.h"

namespace  dldetection{

class EfficientNetWrapperImpl : public torch::nn::Module
{
public:
    EfficientNetWrapperImpl(EfficientNetType type, bool isLoadWeights = false);

    /*!
     * \brief 前向
     * \param x 输入
     * \return
     */
    std::vector<torch::Tensor> forward(torch::Tensor x);

private:



};
TORCH_MODULE(EfficientNetWrapper);

}

#endif // EFFICIENTNETWRAPPER_H
