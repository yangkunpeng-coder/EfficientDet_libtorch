#include "mbconvblock.h"
#include "utils.h"

namespace dldetection {

/*!
 * \brief 构造函数
 * \param blockArgs
 */
MBConvBlockImpl::MBConvBlockImpl(const BlockArgs *blockArgs, const GlobalParams* globalParams)
{
    this->blockArgs = blockArgs;
    this->globalParams = globalParams;

    this->hasSe = 0 < this->blockArgs->seRatio <= 1.0f;

    int64_t inChannel = this->blockArgs->inputFilters;
    int64_t outChannel = this->blockArgs->inputFilters * this->blockArgs->expandRatio;
    if (this->blockArgs->expandRatio != 1)
    {
        this->expandConv = Conv2dStaticSamePadding(inChannel, outChannel, 1, 1, false);
        this->bn0 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(outChannel)\
                                           .momentum(1 - this->globalParams->batchNormMomentum).eps(this->globalParams->batchNormEpsilon));
        register_module("expandConv", this->expandConv);
        register_module("bn0", this->bn0);
    }

    //深度可分离卷积
    this->depthwiseConv = Conv2dStaticSamePadding(outChannel, outChannel, outChannel, \
                                                  this->blockArgs->kernelSize, this->blockArgs->stride, false);

    this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(outChannel)\
                                       .momentum(1 - this->globalParams->batchNormMomentum).eps(this->globalParams->batchNormEpsilon));
    register_module("depthwiseConv", this->depthwiseConv);
    register_module("bn1", this->bn1);

    //压缩和扩张层
    if (this->hasSe)
    {
        int squeezedChannels = std::max(1, static_cast<int>(this->blockArgs->inputFilters * this->blockArgs->seRatio));
        this->seReduce = Conv2dStaticSamePadding(outChannel, squeezedChannels, 1);
        this->seExpand = Conv2dStaticSamePadding(squeezedChannels, outChannel, 1);
        register_module("seReduce", this->seReduce);
        register_module("seExpand", this->seExpand);
    }

    //输出部分
    int finalOutChannel = this->blockArgs->outputFilter;
    this->projectConv = Conv2dStaticSamePadding(outChannel, finalOutChannel, 1, false);
    this->bn2 =torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(finalOutChannel)\
                                      .momentum(1 - this->globalParams->batchNormMomentum).eps(this->globalParams->batchNormEpsilon));
    register_module("projectConv", this->projectConv);
    register_module("bn2", this->bn2);
}

/*!
 * \brief 前向
 * \param x 输入
 * \return
 */
at::Tensor MBConvBlockImpl::forword(at::Tensor inputs, float dropConnectRate)
{
    torch::Tensor x(inputs.clone());
    if (this->blockArgs->expandRatio != 1)
    {
        x = this->expandConv->forward(inputs);
        x = this->bn0->forward(x);
        x = SwishImplementation::apply(x).at(0);
    }

    x = this->depthwiseConv->forward(x);
    x = this->bn1->forward(x);
    x = SwishImplementation::apply(x).at(0);

    //压缩和扩张层
    if(this->hasSe)
    {
        torch::Tensor xSqueezed = torch::nn::functional::adaptive_avg_pool2d(x, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));
        xSqueezed = this->seReduce->forward(x);
        SwishImplementation::apply(x).at(0);
        xSqueezed = this->seExpand->forward(x);
        x = torch::sigmoid(xSqueezed) * x;
    }

    x = this->projectConv->forward(x);
    x = this->bn2->forward(x);

    //跳跃连接和下落连接
    if(this->blockArgs->idSkip && this->blockArgs->stride == 1 && this->blockArgs->inputFilters == this->blockArgs->outputFilter)
    {
        if(0 < dropConnectRate && dropConnectRate < 1.0f)
        {
            x = DropConnect(x, dropConnectRate, this->is_training());
        }
        x = x + inputs;
    }
    return x;
}

}
