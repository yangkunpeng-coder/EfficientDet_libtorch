#include "utils.h"

namespace  dldetection{
/*!
 * \brief 向下连接
 * \param inputs 输入
 * \param p  概率
 * \param isTraining 是否是训练
 * \return
 */
at::Tensor DropConnect(at::Tensor inputs, float p, bool isTraining)
{
    if(!isTraining) return inputs;
    int64_t batcSize = inputs.size(0);
    float keepProb = 1.0f - p;
    torch::Tensor randomTensor = torch::full(inputs.sizes(), at::Scalar(keepProb), at::TensorOptions()\
                                             .dtype(inputs.dtype()).device(inputs.device()));
    randomTensor += torch::rand({batcSize, 1, 1, 1}, at::TensorOptions().dtype(inputs.dtype()).device(inputs.device()));
    torch::Tensor binaryTensor = torch::floor(randomTensor);
    return inputs / keepProb * binaryTensor;
}

/*!
 * \brief 四舍五入通道
 * \param filters 通道
 * \param widthCoefficient 宽度因子
 * \param depthDivisor 深度除数
 * \param minDepth 最小深度
 * \return
 */
int RoundFilters(int filters, float widthCoefficient, int depthDivisor, int minDepth)
{
    float multiplier = widthCoefficient;
    int divisor = depthDivisor;
    int min_depth = minDepth;
    filters *= multiplier;
    if(minDepth == -1)
    {
        min_depth = divisor;
    }
    int new_filters = std::max(min_depth, int(filters + divisor / 2) / divisor * divisor);
    if (new_filters < 0.9 * filters)  // prevent rounding by more than 10%
    {
          new_filters += divisor;
    }
    return new_filters;
}

}


