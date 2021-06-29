#ifndef EFFICIENTDETCONFIG_H
#define EFFICIENTDETCONFIG_H

#include <unordered_map>

#include "efficientdettype.h"

namespace dldetection{

typedef struct EfficientDetLayerConfig
{
    int fpnNumFileter;
    int fpnCellRepeat;
    int inpuSize;
    int boxClassRepeat;
    int pyramidLevel;
    float anchaorScale;
}EfficientDetLayerConfig;

const std::unordered_map<EfficientDetType, EfficientDetLayerConfig> layerConfigMap
{
    {EfficientDetType::EfficientDetB0, EfficientDetLayerConfig{64, 3, 512, 3, 5, 4}},
    {EfficientDetType::EfficientDetB1, EfficientDetLayerConfig{88, 4, 640, 3, 5, 4}},
    {EfficientDetType::EfficientDetB2, EfficientDetLayerConfig{112, 5, 768, 3, 5, 4}},
    {EfficientDetType::EfficientDetB3, EfficientDetLayerConfig{160, 6, 896, 4, 5, 4}},
    {EfficientDetType::EfficientDetB4, EfficientDetLayerConfig{224, 7, 1024, 4, 5, 4}},
    {EfficientDetType::EfficientDetB5, EfficientDetLayerConfig{288, 7, 1280, 4, 5, 4}},
    {EfficientDetType::EfficientDetB6, EfficientDetLayerConfig{384, 8, 1280, 5, 5, 4}},
    {EfficientDetType::EfficientDetB7, EfficientDetLayerConfig{384, 8, 1536, 5, 5, 5}},
    {EfficientDetType::EfficientDetB8, EfficientDetLayerConfig{384, 8, 1536, 5, 6, 4}},
};

}

#endif // EFFICIENTDETCONFIG_H
