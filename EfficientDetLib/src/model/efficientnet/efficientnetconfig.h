#ifndef EFFICIENTNETCONFIG_H
#define EFFICIENTNETCONFIG_H

#include <unordered_map>
#include <vector>

#include "efficientnettype.h"

namespace dldetection {

typedef struct NetConfig
{
    float widthCoefficient;   ///<宽度因子
    float depthCoefficient;   ///<深度因子
    int res;       ///<分辨率
    float dropout; ///<丢弃率
}NetConfig;

const std::unordered_map<EfficientNetType, NetConfig> netConfigMap
{
    {EfficientNetType::EfficientNetB0, NetConfig{1.0f, 1.0f, 224, 0.2f}},
    {EfficientNetType::EfficientNetB1, NetConfig{1.0f, 1.1f, 240, 0.2f}},
    {EfficientNetType::EfficientNetB2, NetConfig{1.1f, 1.2f, 260, 0.3f}},
    {EfficientNetType::EfficientNetB3, NetConfig{1.2f, 1.4f, 300, 0.3f}},
    {EfficientNetType::EfficientNetB4, NetConfig{1.4f, 1.8f, 380, 0.4f}},
    {EfficientNetType::EfficientNetB5, NetConfig{1.6f, 2.2f, 456, 0.4f}},
    {EfficientNetType::EfficientNetB6, NetConfig{1.8f, 2.6f, 528, 0.5f}},
    {EfficientNetType::EfficientNetB7, NetConfig{2.0f, 3.1f, 600, 0.5f}},
    {EfficientNetType::EfficientNetB8, NetConfig{2.2f, 3.6f, 672, 0.5f}},
    {EfficientNetType::EfficientNetL2, NetConfig{4.3f, 5.3f, 800, 0.5f}},
};

typedef struct BlockArgs
{
    int kernelSize;     ///<核大小
    int numRepeat;      ///<重复次数
    int inputFilters;   ///<输入通道
    int outputFilter;   ///<输出通道
    int expandRatio;    ///<扩展率
    bool idSkip;        ///<跳跃层
    int stride;         ///<步长
    float seRatio;      ///<伸缩率
}BlockArgs;

static std::vector<BlockArgs> sblockArgs ///<块参数
{
    {3, 1, 32,  16,  1, true, 1, 0.25},
    {3, 2, 16,  24,  6, true, 2, 0.25},
    {5, 2, 24,  40,  6, true, 2, 0.25},
    {3, 3, 40,  80,  6, true, 2, 0.25},
    {5, 3, 80,  112, 6, true, 1, 0.25},
    {5, 4, 112, 192, 6, true, 2, 0.25},
    {3, 1, 192, 320, 6, true, 1, 0.25},
};

typedef struct GlobalParams
{
    float batchNormMomentum = 0.99f;    ///<批归一化动量
    double batchNormEpsilon = 1e-3;     ///批归一化e
    float dropoutRate;                  ///<丢失率
    float dropConnectRate;              ///<丢失连接率
    // data_format='channels_last',  # removed, this is always true in PyTorch
    int numClasses;                     ///<类别数
    float widthCoefficient;             ///<宽度因子
    float depthCoefficient;             ///<深度因子
    int depthDivisor = 8;               ///<除数
    int minDepth = -1;                  ///<最小深度
    int imageSize;                      ///<图片尺寸
}GlobalParams;

}

#endif // EFFICIENTNETCONFIG_H
