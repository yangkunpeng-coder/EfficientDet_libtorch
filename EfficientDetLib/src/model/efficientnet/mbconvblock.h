#ifndef MBCONVBLOCK_H
#define MBCONVBLOCK_H

#include <vector>

#include "torch/torch.h"
#include "efficientnetconfig.h"
#include "conv2dstaticsamepadding.h"
#include "swishimplementation.h"

namespace dldetection {

/*!
 * \brief Mobile Inverted Residual Bottleneck Block
 */
class MBConvBlockImpl : public torch::nn::Module
{
public:

    /*!
     * \brief 构造函数
     * \param blockArgs
     */
    MBConvBlockImpl(const BlockArgs *blockArgs, const GlobalParams* globalParams);

    /*!
     * \brief 前向
     * \param x 输入
     * \return
     */
    torch::Tensor forword(torch::Tensor x, float dropConnectRate = -1);

private:
    const BlockArgs *blockArgs = nullptr;///<块参数

    const GlobalParams* globalParams = nullptr;///<全局参数

    Conv2dStaticSamePadding expandConv{nullptr};///<扩张卷积

    torch::nn::BatchNorm2d bn0{nullptr};///批归一化

    torch::nn::BatchNorm2d bn1{nullptr};///批归一化

    torch::nn::BatchNorm2d bn2{nullptr};///批归一化

    Conv2dStaticSamePadding depthwiseConv{nullptr};///<深度可分离卷积

    bool hasSe = false;

    Conv2dStaticSamePadding seReduce{nullptr};///<SE缩小

    Conv2dStaticSamePadding seExpand{nullptr};///<SE扩增

    Conv2dStaticSamePadding projectConv{nullptr};///<最后层卷积
};

TORCH_MODULE(MBConvBlock);

}

#endif // MBCONVBLOCK_H
