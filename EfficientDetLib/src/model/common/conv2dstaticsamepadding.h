#ifndef CONV2DSTATICSAMEPADDING_H
#define CONV2DSTATICSAMEPADDING_H

#include "torch/torch.h"

namespace dldetection {

/*!
 * \brief conv2d with same padding
 */
class Conv2dStaticSamePaddingImpl : public torch::nn::Module
{
public:
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
    Conv2dStaticSamePaddingImpl(int64_t inChannel, int64_t outChannel,  torch::ExpandingArray<2, int64_t> kernelSize,\
                            torch::ExpandingArray<2, int64_t> stride = 1, bool bias = true, int groups = 1, int dilation = 0);

    /*!
     * \brief 前向
     * \param x 输入
     * \return
     */
    torch::Tensor forward(torch::Tensor x);

private:

    torch::nn::Conv2d conv{nullptr}; ///<卷积

    torch::ExpandingArray<2, int64_t> strides{}; ///<步长

    torch::ExpandingArray<2, int64_t> kernels{};///<核大小
} ;
TORCH_MODULE(Conv2dStaticSamePadding);
}

#endif // CONV2DSTATICSAMEPADDING_H
