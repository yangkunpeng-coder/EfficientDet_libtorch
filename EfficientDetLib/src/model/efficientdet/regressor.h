#ifndef REGRESSOR_H
#define REGRESSOR_H

#include "torch/torch.h"
#include "separableconvblock.h"

namespace dldetection {

/*!
 * \brief 回归类
 */
class RegressorImpl : public torch::nn::Module
{
public:

    /*!
     * \brief Classifier 分类器
     * \param inChannel 输入通道
     * \param numAnchor anchor数目
     * \param numClass 类别数目
     * \param numLayer 层数
     * \param pyramidLevel 金字塔层数
     * \param onnxExport 导出onnx
     */
    RegressorImpl(int inChannel, int numAnchor, int numLayer, int pyramidLevel = 5, bool onnxExport = false);

    /*!
     * \brief 前向
     * \param x 输入
     * \return
     */
    torch::Tensor forward(std::vector<at::Tensor> inputs);

private:
    int numAnchor;

    int numLayer;

    bool onnxExport;

    torch::nn::ModuleList convList;

    torch::nn::ModuleList bnList;

    SeparableConvBlock header{nullptr};

    MemoryEfficientSwish memorySwish{nullptr};

    Swish swish{nullptr};
};
TORCH_MODULE(Regressor);
}

#endif // REGRESSOR_H
