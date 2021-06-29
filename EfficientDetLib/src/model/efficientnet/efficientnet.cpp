#include <limits>
#include <memory>

#include "efficientnet.h"
#include "utils.h"
#include "swishimplementation.h"
#include "efficientnetconfig.h"

namespace dldetection {

EfficientNet::EfficientNet(EfficientNetType type, int numClass, int inChannels)
{
    this->blockArgs = &sblockArgs;
    GlobalParams sglobalParams;
    sglobalParams.numClasses = numClass;
    sglobalParams.widthCoefficient = netConfigMap.at(type).widthCoefficient;
    sglobalParams.depthCoefficient = netConfigMap.at(type).depthCoefficient;
    sglobalParams.dropConnectRate = netConfigMap.at(type).dropout;
    sglobalParams.imageSize = netConfigMap.at(type).res;
    this->globalParams = &sglobalParams;

    //Stem
    int inChannel = inChannels;//RGB
    int outChannel = RoundFilters(32, this->globalParams->widthCoefficient, this->globalParams->depthDivisor, this->globalParams->minDepth);
    this->convStem = Conv2dStaticSamePadding(inChannel, outChannel, 3, 2, false);
    this->bn0 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(outChannel)\
                                       .momentum(1 - this->globalParams->batchNormMomentum).eps(this->globalParams->batchNormEpsilon));
    register_module("convStem", this->convStem);
    register_module("bn0", this->bn0);
    BlockArgs blockArg;
    for (unsigned int blockArgIndex = 0; blockArgIndex < this->blockArgs->size(); ++blockArgIndex)
    {
        blockArg = blockArgs->at(blockArgIndex);
        blockArg.inputFilters = RoundFilters(blockArg.inputFilters, this->globalParams->widthCoefficient, this->globalParams->depthDivisor);
        blockArg.outputFilter = RoundFilters(blockArg.outputFilter, this->globalParams->widthCoefficient, this->globalParams->depthDivisor);
        blockArg.numRepeat = RoundFilters(blockArg.numRepeat, this->globalParams->widthCoefficient, this->globalParams->depthDivisor);
        this->blocks->push_back(MBConvBlock(&blockArg, this->globalParams));
        if(blockArg.numRepeat > 1)
        {
            blockArg.inputFilters = blockArg.outputFilter;
            blockArg.stride = 1;
        }
        for (int repeatIndex = 0; repeatIndex < blockArg.numRepeat - 1; ++repeatIndex)
        {
            this->blocks->push_back(MBConvBlock(&blockArg, this->globalParams));
        }
    }
    //Head
    inChannel = blockArg.outputFilter;
    outChannel = RoundFilters(1280, this->globalParams->widthCoefficient, this->globalParams->depthDivisor);
    this->convHead = Conv2dStaticSamePadding(inChannel, outChannel, 1, false);
    this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(outChannel)\
                                       .momentum(1 - this->globalParams->batchNormMomentum).eps(this->globalParams->batchNormEpsilon));
    //最后的全连接层
    this->avgPool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1));
    this->drop = torch::nn::Dropout(torch::nn::DropoutOptions(static_cast<double>(this->globalParams->dropoutRate)));
    this->fc = torch::nn::Linear(torch::nn::LinearOptions(outChannel, this->globalParams->numClasses));
}

at::Tensor dldetection::EfficientNet::forword(at::Tensor x)
{
    int64_t batchSize = x.size(0);

    //stem
    x = this->convStem->forward(x);
    x = this->bn0->forward(x);
    x = SwishImplementation::apply(x).at(0);

    //blocks
    for (size_t blockIndex = 0; blockIndex < this->blocks->size(); ++ blockIndex)
    {
        float dropConnectRate = this->globalParams->dropConnectRate;
        if(dropConnectRate > 0 && dropConnectRate < 1.0f)
        {
            dropConnectRate *= static_cast<float>(blockIndex) / this->blocks->size();
        }
        this->blocks[blockIndex]->as<MBConvBlock>()->forword(x, dropConnectRate);
    }

    //Head
    x = this->convHead->forward(x);
    x = this->bn1->forward(x);
    x = SwishImplementation::apply(x).at(0);

    //last layer
    x = this->avgPool->forward(x);
    x = x.view({batchSize, -1});
    x = this->drop->forward(x);
    x = this->fc->forward(x);

    return x;
}

}

