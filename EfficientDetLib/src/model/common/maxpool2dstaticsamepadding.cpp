#include "maxpool2dstaticsamepadding.h"

namespace dldetection {

MaxPool2dStaticSamePaddingImpl::MaxPool2dStaticSamePaddingImpl(torch::ExpandingArray<2, int64_t> kernel, torch::ExpandingArray<2, int64_t> stride)
{
    this->pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(kernel).stride(stride));
    this->kernel = kernel;
    this->stride = stride;

    register_module("pool", this->pool);
}

/*!
 * \brief forward 前向
 * \param x
 * \return
 */
at::Tensor MaxPool2dStaticSamePaddingImpl::forward(at::Tensor x)
{
    int64_t width = x.size(-1);
    int64_t height = x.size(-2);

    double extraH = (std::ceil(width / this->stride->at(1)) - 1) * this->stride->at(1) - width + this->kernel->at(1);
    double extraV = (std::ceil(height / this->stride->at(0)) - 1) * this->stride->at(0) - height + this->kernel->at(0);

    int left = static_cast<int>(extraH) / 2;
    int right = static_cast<int>(extraV - left);
    int top = static_cast<int>(extraV) / 2;
    int bottom = static_cast<int>(extraV - top);

    std::vector<int64_t> pads {left, right, top, bottom};
    torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions(pads));

    x = this->pool->forward(x);

    return x;
}

}

