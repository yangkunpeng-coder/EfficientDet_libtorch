#ifndef MAXPOOL2DSTATICSAMEPADDING_H
#define MAXPOOL2DSTATICSAMEPADDING_H

#include "torch/torch.h"

namespace dldetection {

/*!
 * \brief 相同padding最大池化MaxPool2d
 */
class MaxPool2dStaticSamePaddingImpl : public torch::nn::Module
{
public:

    MaxPool2dStaticSamePaddingImpl(torch::ExpandingArray<2, int64_t> kernel, torch::ExpandingArray<2, int64_t> stride);

    /*!
     * \brief forward 前向
     * \param x
     * \return
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::MaxPool2d pool{nullptr};

    torch::ExpandingArray<2, int64_t> stride{};

    torch::ExpandingArray<2, int64_t> kernel{};

};

TORCH_MODULE(MaxPool2dStaticSamePadding);
}


#endif // MAXPOOL2DSTATICSAMEPADDING_H
