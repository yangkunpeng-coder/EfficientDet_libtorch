#ifndef SEPARABLECONVBLOCK_H
#define SEPARABLECONVBLOCK_H

#include "torch/torch.h"

#include "conv2dstaticsamepadding.h"
#include "swish.h"
#include "memoryefficientswish.h"

namespace dldetection {

/*!
 * \brief 可分离卷积
 */
class SeparableConvBlockImpl : public torch::nn::Module
{
public:

    /*!
     * \brief SeparableConvBlock 可分离卷积
     * \param inChannel 输入通道
     * \param outChanels 输出通道
     * \param norm 是否Norm
     * \param activation 激活
     * \param onnxExport onnx导出
     *  note:! pointwise需要bias，depthwise 没有bias
     */
    SeparableConvBlockImpl(int inChannel, int outChanels, bool norm = true, bool activation = false, bool onnxExport = false);

    /*!
     * \brief 前向
     * \param x 输入
     * \return
     */
    torch::Tensor forward(torch::Tensor x);


private:

    Conv2dStaticSamePadding depthWiseConv{nullptr};///<深度卷积

    Conv2dStaticSamePadding pointWiseConv{nullptr};///<点卷积

    torch::nn::BatchNorm2d bn{nullptr};///<批归一化 pt_momentum = 1 - tf_momentum

    bool norm = true;

    bool activation = false;

    bool onnxExport = false;

    Swish activate{nullptr};///<激活函数

    MemoryEfficientSwish memoryActivate{nullptr};///<激活函数
};

TORCH_MODULE(SeparableConvBlock);
}

#endif // SEPARABLECONVBLOCK_H
