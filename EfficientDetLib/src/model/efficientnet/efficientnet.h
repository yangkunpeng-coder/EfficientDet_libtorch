#ifndef EFFICIENTNET_H
#define EFFICIENTNET_H

#include <vector>

#include "torch/torch.h"

#include "efficientnetconfig.h"
#include "conv2dstaticsamepadding.h"
#include "mbconvblock.h"

namespace dldetection {

/*!
 * \brief efficientnet 模型
 */
class EfficientNet : public torch::nn::Module
{
public:

    /*!
     * \brief EfficientNet 获取模型
     * \param type 模型类型
     * \param numClass 类别数
     * \param inChannels 通道数
     */
    EfficientNet(EfficientNetType type, int numClass, int inChannels);

    /*!
     * \brief 前向
     * \param x 输入
     * \return
     */
    torch::Tensor forword(torch::Tensor x);

private:

    std::vector<BlockArgs> *blockArgs = nullptr;///<块参数

    const GlobalParams* globalParams = nullptr;///<全局参数

    Conv2dStaticSamePadding convStem {nullptr};///<Stem

    torch::nn::BatchNorm2d bn0{nullptr};///<bn0

    torch::nn::ModuleList blocks{};///<重复的块

    Conv2dStaticSamePadding convHead{nullptr};///<头部卷积

    torch::nn::BatchNorm2d bn1 {nullptr};///<批归一化

    torch::nn::AdaptiveAvgPool2d avgPool{nullptr};///<平均池化

    torch::nn::Dropout drop{nullptr};///<丢弃

    torch::nn::Linear fc{nullptr};///<全连接层
};

}


#endif // EFFICIENTNET_H
