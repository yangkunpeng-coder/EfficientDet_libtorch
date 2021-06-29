#ifndef ANCHORS_H
#define ANCHORS_H

#include <unordered_map>

#include "torch/torch.h"
#include "utils.h"

namespace dldetection {

/*!
 * \brief 产生anchor box
 */
class Anchors : public torch::nn::Module
{
public:

    Anchors(double anchorScale = 4, std::vector<int> pyramidLevels = {3, 4, 5, 6, 7}, \
            std::vector<int> strides = {}, std::vector<double> scales = {}, std::vector<std::tuple<double, double>> ratiod = {});

    /*!
     * \brief
     * \param image 图片
     * \param dtype 数据类型
     * \return
     */
    torch::Tensor forward(torch::Tensor image, torch::ScalarType dtype = torch::kF32);

private:
    double anchorScale;

    std::vector<int> pyramidLevels;

    std::vector<int> strides = {};

    std::vector<double> scales = {};

    std::vector<std::tuple<double, double>> ratios = {};

    std::unordered_map<c10::Device, torch::Tensor> lastAnchors;
};

}

#endif // ANCHORS_H
