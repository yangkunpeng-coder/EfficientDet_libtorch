#ifndef UTILS_H
#define UTILS_H

#include "torch/torch.h"

namespace dldetection {
/*!
 * \brief 向下连接
 * \param inputs 输入
 * \param p  概率
 * \param isTraining 是否是训练
 * \return
 */
torch::Tensor DropConnect(torch::Tensor inputs, float p, bool isTraining);

/*!
 * \brief 四舍五入通道
 * \param filters 通道
 * \param widthCoefficient 宽度因子
 * \param depthDivisor 深度除数
 * \param minDepth 最小深度
 * \return
 */
int RoundFilters(int filters, float widthCoefficient, int depthDivisor, int minDepth = -1);

/*!
 * \brief CartesianProduct  笛卡尔积
 * \param A 集合A
 * \param B 集合B
 * \return 元祖
 */
template<typename SetT1, typename SetT2>
std::vector<std::tuple<SetT1, SetT2> > CartesianProduct(const std::vector<SetT1> &A, const std::vector<SetT2> &B)
{
    std::vector<std::tuple<SetT1, SetT2> > product;
    for (size_t i = 0; i < A.size(); ++i)
    {
        for (size_t j = 0; j < B.size(); ++j)
        {
            product.push_back(std::make_tuple(A[i], B[j]));
        }
    }
    return product;
}

/*!
 * \brief instanceof 判断是否是父类
 * \return
 */
template<typename Base, typename T>
inline bool InstanceOf(const T*)
{
   return std::is_base_of<Base, T>::value;
}


}

#endif // UTILS_H
