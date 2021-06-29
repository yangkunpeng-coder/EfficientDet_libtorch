#ifndef MEMORYEFFICIENTSWISH_H
#define MEMORYEFFICIENTSWISH_H

#include "torch/torch.h"
#include "swishimplementation.h"

namespace dldetection {

/*!
 * \brief 激活函数，优化了内存
 */
class MemoryEfficientSwishImpl : public torch::nn::Module
{
public:
    MemoryEfficientSwishImpl();

    /*!
     * \brief forward 前向
     * \param x
     * \return
     */
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MemoryEfficientSwish);

}

#endif // MEMORYEFFICIENTSWISH_H
