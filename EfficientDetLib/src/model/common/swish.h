#ifndef SWISH_H
#define SWISH_H

#include "torch/torch.h"

namespace dldetection {

class SwishImpl : public torch::nn::Module
{
public:
    SwishImpl();

    /*!
     * \brief forward 前向
     * \param x
     * \return
     */
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Swish);

}


#endif // SWISH_H
