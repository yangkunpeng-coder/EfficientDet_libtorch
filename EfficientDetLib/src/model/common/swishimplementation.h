#ifndef SWISHIMPLEMENTATION_H
#define SWISHIMPLEMENTATION_H

#include "torch/torch.h"

namespace dldetection {

/*!
 * \brief 自定义自动计算反向梯度算子,相比其他激活函数，更加高效
 *  参考 https://pytorch.org/cppdocs/api
 */
class SwishImplementation : public torch::autograd::Function<SwishImplementation>
{
public:
    /*!
     * \brief 前向计算
     * \param ctx 上下文
     * \param i 计算梯度的矩阵
     * \return
     */
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx, torch::Tensor i)
    {
        //在context中为后向传播保存数据
        torch::Tensor result = torch::sigmoid(i);
        ctx->saved_data["i"] = i;
        ctx->mark_dirty({i});
        return {i};
    }

    /*!
     * \brief 回传
     * \param ctx
     * \param gradOutput  梯度输出
     * \return
     */
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list gradOutput)
    {
        //在前向中使用数据保存
        torch::Tensor i = ctx->saved_data["i"].toTensor();
        torch::Tensor sigmoid_i = torch::sigmoid(i);
        return {gradOutput[0] * (sigmoid_i * (1 + i * (1 - sigmoid_i)))};
    }
};

}

#endif // SWISHIMPLEMENTATION_H
