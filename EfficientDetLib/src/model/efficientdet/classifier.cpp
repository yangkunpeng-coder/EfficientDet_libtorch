#include "classifier.h"

namespace dldetection {

/*!
 * \brief Classifier 分类器
 * \param inChannel 输入通道
 * \param numAnchor anchor数目
 * \param numClass 类别数目
 * \param numLayer 层数
 * \param pyramidLevel 金字塔层数
 * \param onnxExport 导出onnx
 */
ClassifierImpl::ClassifierImpl(int inChannel, int numAnchor,int numClass, \
                       int numLayer, int pyramidLevel, bool onnxExport)
{
    this->numAnchor = numAnchor;
    this->numClass = numClass;
    this->numLayer = numLayer;
    this->onnxExport = onnxExport;

    for (int layerIndex = 0; layerIndex < numLayer; ++layerIndex)
    {
         this->convList->push_back(SeparableConvBlock(inChannel, inChannel, false));
    }

    for (int levelIndex = 0; levelIndex < pyramidLevel; ++levelIndex)
    {
        torch::nn::ModuleList levelList;
        for (int layerIndex = 0; layerIndex < numLayer; ++layerIndex)
        {
            levelList->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(inChannel).momentum(0.01).eps(1e-3)));
        }
        this->bnList->push_back(levelList);
    }

    this->header = SeparableConvBlock(inChannel, numAnchor * numClass, false, false);

    if(!onnxExport)
    {
        this->memorySwish = MemoryEfficientSwish();
    }
    else
    {
        this->swish = Swish();
    }

    register_module("convList", this->convList);
    register_module("bnList", this->bnList);
    register_module("header", this->header);
    if(!onnxExport)
    {
        register_module("memorySwish", this->memorySwish);
    }
    else
    {
        register_module("swish", this->swish);
    }
}

/*!
 * \brief 前向
 * \param x 输入
 * \return
 */
at::Tensor ClassifierImpl::forward(std::vector<torch::Tensor> inputs)
{
    torch::TensorList outputs;
    for (size_t pyramidLevelIndex = 0; pyramidLevelIndex < inputs.size(); ++pyramidLevelIndex)
    {
        torch::Tensor output;
        for(int layerIndex = 0; layerIndex < this->numLayer; ++layerIndex)
        {
            output = this->convList[layerIndex]->as<SeparableConvBlock>()->forward(inputs.at(pyramidLevelIndex));
            output = this->bnList[pyramidLevelIndex]->as<torch::nn::ModuleList>()[layerIndex].as<torch::nn::BatchNorm2d>()->forward(output);
            if(!this->onnxExport)
            {
                output = this->memorySwish->forward(output);
            }
            else
            {
                output = this->swish->forward(output);
            }
        }
        output = this->header->forward(output);

        output = output.permute({0, 2, 3, 1});

        output = output.contiguous().view({output.size(0), output.size(1), output.size(2), this->numAnchor, this->numClass});

        output = output.contiguous().view({output.size(0), -1, this->numClass});

        outputs.vec().push_back(output);
    }

    torch::Tensor outTensor = torch::cat(outputs, 1);
    outTensor = outTensor.sigmoid();
    return outTensor;
}

}
