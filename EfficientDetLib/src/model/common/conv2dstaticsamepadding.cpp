#include "conv2dstaticsamepadding.h"

#include <math.h>


namespace dldetection {

/*!
 * \brief  构造函数
 * \param inChannel 输入通道
 * \param outChannel 输出通道
 * \param kernelSize 核大小
 * \param stride 步长
 * \param bias 偏置项
 * \param groups 分组
 * \param dilation 空洞
 */
Conv2dStaticSamePaddingImpl::Conv2dStaticSamePaddingImpl(int64_t inChannel, int64_t outChannel,  torch::ExpandingArray<2, int64_t> kernelSize, \
                                                  torch::ExpandingArray<2, int64_t> stride, bool bias, int groups, int dilation)
{
    this->strides = stride;
    this->kernels = kernelSize;
    this->conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(inChannel, outChannel, kernelSize)\
                                   .stride(stride).bias(bias).groups(groups).dilation(dilation));
    register_module("conv", this->conv);
}

/*!
 * \brief 前向
 * \param x 输入
 * \return
 */
at::Tensor Conv2dStaticSamePaddingImpl::forward(at::Tensor x)
{
    int64_t width = x.size(-1);
    int64_t height = x.size(-2);

    double extraH = (std::ceil(width / this->strides->at(1)) - 1) * this->strides->at(1) - width + this->kernels->at(1);
    double extraV = (std::ceil(height / this->strides->at(0)) - 1) * this->strides->at(0) - height + this->kernels->at(0);

    int left = static_cast<int>(extraH) / 2;
    int right = static_cast<int>(extraV - left);
    int top = static_cast<int>(extraV) / 2;
    int bottom = static_cast<int>(extraV - top);
    std::vector<int64_t> pads {left, right, top, bottom};
    torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions(pads));

    x = this->conv->forward(x);

    return x;
}

}

