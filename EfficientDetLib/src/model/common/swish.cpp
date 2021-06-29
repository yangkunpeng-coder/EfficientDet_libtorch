#include "swish.h"

namespace dldetection {

SwishImpl::SwishImpl()
{

}

/*!
 * \brief forward 前向
 * \param x
 * \return
 */
at::Tensor SwishImpl::forward(at::Tensor x)
{
    return x * torch::sigmoid(x);
}

}
