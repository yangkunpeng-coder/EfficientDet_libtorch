#include "anchors.h"

namespace dldetection {

Anchors::Anchors(double anchorScale, std::vector<int> pyramidLevels, \
                 std::vector<int> strides, std::vector<double> scales, std::vector<std::tuple<double, double>> ratiod )
{
    this->anchorScale = anchorScale;
    this->pyramidLevels = pyramidLevels;

    if(strides.empty())
    {
        for (size_t i = 0; i < pyramidLevels.size(); ++i)
        {
            this->strides.push_back(2 ^ pyramidLevels[i]);
        }
    }
    else this->strides = strides;

    if(scales.empty())
    {
        this->scales.push_back(std::pow(2, 0));
        this->scales.push_back(std::pow(2, (1.0 / 3.0)));
        this->scales.push_back(std::pow(2, (2.0 / 3.0)));
    }
    else this->scales = scales;

    if(ratiod.empty())
    {
        this->ratios.push_back(std::make_tuple(1.0, 1.0));
        this->ratios.push_back(std::make_tuple(1.4, 0.7));
        this->ratios.push_back(std::make_tuple(0.7, 1.4));
    }
    else this->ratios = ratiod;
}

at::Tensor Anchors::forward(at::Tensor image, c10::ScalarType dtype)
{
    std::tuple<int, int> imageShape; ///<高和宽

    imageShape = std::make_tuple(image.size(2), image.size(3));
    std::vector<torch::Tensor> boxesAll;

    for (size_t strideIndex = 0; strideIndex < this->strides.size(); ++strideIndex)
    {
        int stride = this->strides[strideIndex];
        std::vector<torch::Tensor> boxesLevel;
        std::vector<std::tuple<double, std::tuple<double, double>>> scaleAndRatio;
        scaleAndRatio = CartesianProduct(this->scales, this->ratios);
        for (size_t i = 0; i < scaleAndRatio.size(); ++i)
        {
            double scale = std::get<0>(scaleAndRatio[i]);
            std::tuple<double, double> ratio = std::get<1>(scaleAndRatio[i]);
            if(std::get<1>(imageShape) % this->strides[strideIndex] != 0)
            {
                std::cout << "Error : from at::Tensor Anchors::forward(at::Tensor image, c10::ScalarType dtype)" << std::endl;
            }
            double baseAnchorSize = this->anchorScale * stride * scale;
            double anchorSizeX2 = baseAnchorSize * std::get<0>(ratio) / 2.0;
            double anchorSizeY2 = baseAnchorSize * std::get<1>(ratio) / 2.0;
                        
            torch::Tensor x = torch::arange(stride / 2, std::get<1>(imageShape), stride);
            torch::Tensor y = torch::arange(stride / 2, std::get<0>(imageShape), stride);
            std::vector<torch::Tensor> yVxV = torch::meshgrid({x, y});
            torch::Tensor xv = yVxV[1].reshape(-1);
            torch::Tensor yv = yVxV[0].reshape(-1);

            //y1 x1 y2 x2
            torch::Tensor boxes = torch::vstack({yv - anchorSizeY2, xv - anchorSizeX2, yv + anchorSizeY2, xv + anchorSizeX2});
            boxes = torch::swapaxes(boxes, 0, 1);
            boxesLevel.push_back(torch::unsqueeze(boxes, 1));
        }
        //concat anchors on the same level to the reshape NxAx4
        torch::Tensor boxesLevelCat = torch::cat(boxesLevel, 1);
        boxesAll.push_back(boxesLevelCat);
    }
    torch::Tensor anchorBoxes = torch::vstack(boxesAll);
    anchorBoxes.toType(dtype).to(image.device());
    anchorBoxes = anchorBoxes.unsqueeze(0);

    //save it for later use to reduce overhead
    lastAnchors.insert(std::pair<c10::Device, torch::Tensor>(image.device(), anchorBoxes));
    return anchorBoxes;
}

}

