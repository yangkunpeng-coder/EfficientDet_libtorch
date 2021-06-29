#include "memoryefficientswish.h"

namespace dldetection {



MemoryEfficientSwishImpl::MemoryEfficientSwishImpl()
{

}

/*!
 * \brief forward 前向
 * \param x
 * \return
 */
at::Tensor MemoryEfficientSwishImpl::forward(at::Tensor x)
{
    return SwishImplementation::apply(x)[0];
}

}

