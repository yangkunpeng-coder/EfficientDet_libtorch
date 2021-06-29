#include "separableconvblock.h"

namespace dldetection {

/*!     Conv2dStaticSamePaddingImpl(int64_t inChannel, int64_t outChannel,  torch::ExpandingArray<2, int64_t> kernelSize,\
                            torch::ExpandingArray<2, int64_t> stride = 1, bool bias = true, int groups = 1, int dilation = 0);
 * \brief SeparableConvBlock 可分离卷积
 * \param inChannel 输入通道
 * \param outChanels 输出通道
 * \param norm 是否Norm
 * \param activation 激活
 * \param onnxExport onnx导出
 *  note:! pointwise需要bias，depthwise 没有bias
 */
SeparableConvBlockImpl::SeparableConvBlockImpl(int inChannel, int outChanel, bool norm, bool activation, bool onnxExport)
{
    this->depthWiseConv = Conv2dStaticSamePadding(inChannel, inChannel, 3, 1, inChannel, false);
    this->pointWiseConv = Conv2dStaticSamePadding(inChannel, outChanel, 1, 1);

    this->norm = norm;
    if(norm)
    {
        this->bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(outChanel).momentum(0.01).eps(1e-3));
    }
    this->activation = activation;
    this->onnxExport = onnxExport;

    if(activation)
    {
        if(!onnxExport)
        {
            this->memoryActivate = MemoryEfficientSwish();
        }
        else
        {
            this->activate = Swish();
        }
    }

    register_module("depthWiseConv", this->depthWiseConv);
    register_module("pointWiseConv", this->pointWiseConv);
    if(norm)
    {
        register_module("bn", this->bn);
    }
    if(activation)
    {
        if(!onnxExport)
        {
            register_module("memoryActivate", this->memoryActivate);
        }
        else
        {
            register_module("activate", this->activate);
        }
    }
}

/*!
 * \brief 前向
 * \param x 输入
 * \return
 */
at::Tensor SeparableConvBlockImpl::forward(at::Tensor x)
{
    x = this->depthWiseConv->forward(x);
    x = this->pointWiseConv->forward(x);

    if(this->norm)
    {
        x = this->bn->forward(x);
    }
    if(activation)
    {
        if(!onnxExport)
        {
            x = this->memoryActivate->forward(x);
        }
        else
        {
            x = this->activate->forward(x);
        }
    }
    return x;
}

}

