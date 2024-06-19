#ifndef __YOLOV8_COMMON_H__
#define __YOLOV8_COMMON_H__

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <string>
#include "NvInfer.h"

#include "cv_utils.h"
#include "trt_utils.h"
#include "ModelCommon.h"
#include "ModelConfig.h"

namespace NAMESPACE_YOLOv10
{
    #define DEVICE 0  // GPU id
    
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f; ///< [KSAI] IGNORE_THRESH与MAX_OUTPUT_BBOX_COUNT相互作用，
                                                  ///< 当IGNORE_THRESH值过小时，会提前筛选够MAX_OUTPUT_BBOX_COUNT个结果，导致漏掉某些有效目标
    struct YoloKernel{
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 100;

    IScaleLayer *addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, float eps);

    ILayer *convBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int ksize, int s, int g, std::string lname, bool act = true);

    ILayer *C2f(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int n, bool shortcut, int g, float e, std::string lname);

    ILayer *C2fCIB(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int n, bool shortcut, bool lk, int g, float e, std::string lname);

    ILayer *focus(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int inch, int outch, int ksize, std::string lname);

    ILayer *bottleneck(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, bool shortcut, int g, float e, std::string lname);

    ILayer *CIB(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c2, bool shortcut, int g, float e, bool lk, std::string lname);

    ILayer *SCDown(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c2, int k, int s, std::string lname);

    ILayer *bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname);

    ILayer *C3(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname);

    ILayer *SPP(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int k1, int k2, int k3, std::string lname);

    ILayer *SPPF(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int k, std::string lname);

    IPluginV2Layer *addDetectHead(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, std::string lname, std::vector<IConvolutionLayer *> dets);

    ILayer *Attention(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int dim, int num_heads, float attn_ratio, std::string lname);
    ILayer *PSA(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, float e, std::string lname);

}

#endif ///< __YOLOV8_COMMON_H__
