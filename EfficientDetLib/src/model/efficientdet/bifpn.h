#ifndef BIFPN_H
#define BIFPN_H

#include "torch/torch.h"

#include "separableconvblock.h"
#include "maxpool2dstaticsamepadding.h"
#include "memoryefficientswish.h"

#include <vector>

namespace dldetection {

/*!
 * \brief 特征金字塔结构
 */
class BiFPNImpl : public torch::nn::Module
{
public:

    /*!
     * \brief BiFPN 积木
     * \param channelNum 通道数
     * \param convChannels 卷积通道列表
     * \param firstTime 是否第一次
     * \param epsilon
     * \param onnxExport 是否导出ONNX
     * \param attention 是否加注意力
     * \param useP8 是否使用P8层
     */
    BiFPNImpl(int channelNum, std::vector<int> convChannels, bool firstTime, \
          double epsilon = 1e-4, bool onnxExport = false, bool attention = true, bool useP8 = false);

    /*!
     * \brief 前向
     * \param x 输入
     * \return
     */
    std::vector<torch::Tensor> forward(std::vector<torch::Tensor> inputs);

private:

    double epsilon = 1e-4;

    bool useP8 = false;

    bool attention = true;

    bool firstTime;

    bool onnxExport;


    //深度可分离卷积
    SeparableConvBlock conv6Up{nullptr};

    SeparableConvBlock conv5Up{nullptr};

    SeparableConvBlock conv4Up{nullptr};

    SeparableConvBlock conv3Up{nullptr};

    SeparableConvBlock conv4Down{nullptr};

    SeparableConvBlock conv5Down{nullptr};

    SeparableConvBlock conv6Down{nullptr};

    SeparableConvBlock conv7Down{nullptr};

    //上采样
    torch::nn::Upsample p6UpSample{nullptr};

    torch::nn::Upsample p5UpSample{nullptr};

    torch::nn::Upsample p4UpSample{nullptr};

    torch::nn::Upsample p3UpSample{nullptr};


    //最大池化
    MaxPool2dStaticSamePadding p4DownSample{nullptr};

    MaxPool2dStaticSamePadding p5DownSample{nullptr};

    MaxPool2dStaticSamePadding p6DownSample{nullptr};

    MaxPool2dStaticSamePadding p7DownSample{nullptr};

    //激活
    MemoryEfficientSwish memorySwish{nullptr};

    Swish swish{nullptr};

    //if first time
    torch::nn::Sequential p5DownChannel{nullptr};

    torch::nn::Sequential p4DownChannel{nullptr};

    torch::nn::Sequential p3DownChannel{nullptr};

    torch::nn::Sequential p5ToP6{nullptr};

    torch::nn::Sequential p6ToP7{nullptr};

    torch::nn::Sequential p7ToP8{nullptr};

    torch::nn::Sequential p4DownChannel2{nullptr};

    torch::nn::Sequential p5DownChannel2{nullptr};

    //权重

};

TORCH_MODULE(BiFPN);

}

#endif // BIFPN_H
