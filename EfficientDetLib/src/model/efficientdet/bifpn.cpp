#include "bifpn.h"

namespace dldetection {

BiFPNImpl::BiFPNImpl(int channelNum, std::vector<int> convChannels, bool firstTime, \
             double epsilon, bool onnxExport, bool attention, bool useP8)
{
    this->epsilon = epsilon;
    this->useP8 = useP8;
    this->attention = attention;
    this->onnxExport = onnxExport;

    //卷积层
    this->conv6Up = SeparableConvBlock(channelNum, channelNum, true, false, onnxExport);
    this->conv5Up = SeparableConvBlock(channelNum, channelNum, true, false, onnxExport);
    this->conv4Up = SeparableConvBlock(channelNum, channelNum, true, false, onnxExport);
    this->conv3Up = SeparableConvBlock(channelNum, channelNum, true, false, onnxExport);

    this->conv4Down = SeparableConvBlock(channelNum, channelNum, true, false, onnxExport);
    this->conv5Down = SeparableConvBlock(channelNum, channelNum, true, false, onnxExport);
    this->conv6Down = SeparableConvBlock(channelNum, channelNum, true, false, onnxExport);
    this->conv7Down = SeparableConvBlock(channelNum, channelNum, true, false, onnxExport);

    //特征尺度
    this->p6UpSample = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2, 2}).mode(torch::kNearest));
    this->p5UpSample = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2, 2}).mode(torch::kNearest));
    this->p4UpSample = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2, 2}).mode(torch::kNearest));
    this->p3UpSample = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>{2, 2}).mode(torch::kNearest));

    this->p4DownSample = MaxPool2dStaticSamePadding(3, 2);
    this->p5DownSample = MaxPool2dStaticSamePadding(3, 2);
    this->p6DownSample = MaxPool2dStaticSamePadding(3, 2);
    this->p7DownSample = MaxPool2dStaticSamePadding(3, 2);

    if(!onnxExport)
    {
        this->memorySwish = MemoryEfficientSwish();
    }
    else
    {
        this->swish = Swish();
    }

    this->firstTime = firstTime; //第一次进入BIFPN需要特殊处理

    this->p5DownChannel->push_back(Conv2dStaticSamePadding(convChannels.at(2), channelNum, 1));
    this->p5DownChannel->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channelNum).momentum(0.01).eps(1e-3)));

    this->p4DownChannel->push_back(Conv2dStaticSamePadding(convChannels.at(1), channelNum, 1));
    this->p4DownChannel->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channelNum).momentum(0.01).eps(1e-3)));

    this->p3DownChannel->push_back(Conv2dStaticSamePadding(convChannels.at(0), channelNum, 1));
    this->p3DownChannel->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channelNum).momentum(0.01).eps(1e-3)));

    this->p5ToP6->push_back(Conv2dStaticSamePadding(convChannels.at(2), channelNum, 1));
    this->p5ToP6->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channelNum).momentum(0.01).eps(1e-3)));
    this->p5ToP6->push_back(MaxPool2dStaticSamePadding(3, 2));

    this->p6ToP7->push_back(MaxPool2dStaticSamePadding(3, 2));

    this->p4DownChannel2->push_back(Conv2dStaticSamePadding(convChannels.at(1), channelNum, 1));
    this->p4DownChannel2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channelNum).momentum(0.01).eps(1e-3)));

    this->p5DownChannel2->push_back(Conv2dStaticSamePadding(convChannels.at(2), channelNum, 1));
    this->p5DownChannel2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channelNum).momentum(0.01).eps(1e-3)));

    register_module("conv6Up", this->conv6Up);
    register_module("conv5Up", this->conv5Up);
    register_module("conv4Up", this->conv4Up);
    register_module("conv3Up", this->conv3Up);
    register_module("conv4Down", this->conv4Down);
    register_module("conv5Down", this->conv5Down);
    register_module("conv6Down", this->conv6Down);
    register_module("conv7Down", this->conv7Down);
    register_module("p6UpSample", this->p6UpSample);
    register_module("p5UpSample", this->p5UpSample);
    register_module("p4UpSample", this->p4UpSample);
    register_module("p3UpSample", this->p3UpSample);
    register_module("p4DownSample", this->p4DownSample);
    register_module("p5DownSample", this->p5DownSample);
    register_module("p6DownSample", this->p6DownSample);
    register_module("p7DownSample", this->p7DownSample);
    if(!onnxExport)
    {
        register_module("memorySwish", this->memorySwish);
    }
    else
    {
        register_module("swish", this->swish);
    }
    register_module("p5DownChannel", this->p5DownChannel);
    register_module("p4DownChannel", this->p4DownChannel);
    register_module("p3DownChannel", this->p3DownChannel);
    register_module("p5ToP6", this->p5ToP6);
    register_module("p6ToP7", this->p6ToP7);
    register_module("p4DownChannel2", this->p4DownChannel2);
    register_module("p5DownChannel2", this->p5DownChannel2);
}

/*!
 * \brief 前向
 * \param x 输入
 * illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
 * \return
 */
std::vector<torch::Tensor> BiFPNImpl::forward(std::vector<at::Tensor> inputs)
{
    torch::Tensor p3In, p4In, p5In, p6In, p7In, p8In;
    torch::Tensor p3, p4, p5;
    torch::Tensor p4Up, p5Up, p6Up;
    torch::Tensor p3Out, p4Out, p5Out, p6Out, p7Out;

    if(firstTime)
    {
        p3 = inputs.at(0);
        p4 = inputs.at(1);
        p5 = inputs.at(2);

        p6In = this->p5ToP6->forward(p5);
        p7In = this->p6ToP7->forward(p6In);
        p3In = this->p3DownChannel->forward(p3);
        p4In = this->p4DownChannel->forward(p4);
        p5In = this->p5DownChannel->forward(p5);
    }
    else
    {
        p3In = inputs.at(0);
        p4In = inputs.at(1);
        p5In = inputs.at(2);
        p6In = inputs.at(3);
        p7In = inputs.at(4);
    }

    if(!onnxExport)
    {
        p6Up = this->conv6Up(this->memorySwish->forward(p6In + this->p6UpSample->forward(p7In)));
        p5Up = this->conv5Up(this->memorySwish->forward(p5In + this->p5UpSample->forward(p6Up)));
        p4Up = this->conv4Up(this->memorySwish->forward(p4In + this->p4UpSample->forward(p5Up)));
        p3Out = this->conv3Up(this->memorySwish->forward(p3In + this->p3UpSample->forward(p4Up)));
    }
    else
    {
        p6Up = this->conv6Up(this->swish->forward(p6In + this->p6UpSample->forward(p7In)));
        p5Up = this->conv5Up(this->swish->forward(p5In + this->p5UpSample->forward(p6Up)));
        p4Up = this->conv4Up(this->swish->forward(p4In + this->p4UpSample->forward(p5Up)));
        p3Out = this->conv3Up(this->swish->forward(p3In + this->p3UpSample->forward(p4Up)));
    }


    if(this->firstTime)
    {
        p4In = this->p4DownChannel2->forward(p4);
        p5In = this->p5DownChannel2->forward(p5);
    }

    if(!onnxExport)
    {
        p4Out = this->conv4Down->forward(this->memorySwish->forward(p4In + p4Up + this->p4DownSample->forward(p3Out)));
        p5Out = this->conv5Down->forward(this->memorySwish->forward(p5In + p5Up + this->p5DownSample->forward(p4Out)));
        p6Out = this->conv6Down->forward(this->memorySwish->forward(p6In + p6Up + this->p6DownSample->forward(p5Out)));
        p7Out = this->conv7Down->forward(this->memorySwish->forward(p7In + this->p7DownSample->forward(p6Out)));
    }
    else
    {
        p4Out = this->conv4Down->forward(this->swish->forward(p4In + p4Up + this->p4DownSample->forward(p3Out)));
        p5Out = this->conv5Down->forward(this->swish->forward(p5In + p5Up + this->p5DownSample->forward(p4Out)));
        p6Out = this->conv6Down->forward(this->swish->forward(p6In + p6Up + this->p6DownSample->forward(p5Out)));
        p7Out = this->conv7Down->forward(this->swish->forward(p7In + this->p7DownSample->forward(p6Out)));
    }

    return {p3Out, p4Out, p5Out, p6Out, p7Out};
}

}
