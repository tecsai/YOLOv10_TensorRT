#include <assert.h>
#include "YOLOv10_Common.h"

namespace NAMESPACE_YOLOv10
{
IScaleLayer *addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, float eps)
{
    float *gamma = (float *)weightMap[lname + ".weight"].values;
    float *beta = (float *)weightMap[lname + ".bias"].values;
    float *mean = (float *)weightMap[lname + ".running_mean"].values;
    float *var = (float *)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++)
    {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer *convBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int ksize, int s, int g, std::string lname, bool act)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = ksize / 3;
    IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setName((lname+".conv").data());
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);
    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);
    bn1->setName((lname+".bn").data());

    if(!act){
        return bn1;
    }
#if 0
    // Leaky_relu
    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(lr);
    lr->setAlpha(0.1);
    return lr;

#else
    // silu = x * sigmoid
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
#endif
}

ILayer *C2f(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int n, bool shortcut, int g, float e, std::string lname)
{
    ILayer* aggregate_layers[10];
    ITensor* aggregate_tensors[10];
    int last_layer_idx = 0;
    if(n>8){
        perror("Bottleneck counts axceed buffer, should not be more than 8\n");
        exit(0);
    }
    int c = int(outch*e);
    auto cv1 = convBlock(network, weightMap, input, c*2, 1, 1, 1, lname + ".cv1");
    Dims cv1_dim = cv1->getOutput(0)->getDimensions();

    ISliceLayer *s1 = network->addSlice(*cv1->getOutput(0), Dims{4, {0, 0, 0, 0}}, 
                                                            Dims{4, {cv1_dim.d[0], cv1_dim.d[1]/2, cv1_dim.d[2], cv1_dim.d[3]}}, 
                                                            Dims{4, {1, 1, 1, 1}});
    s1->setName(std::string(lname + ".s1").data());
    ISliceLayer *s2 = network->addSlice(*cv1->getOutput(0), Dims{4, {0, cv1_dim.d[1]/2, 0, 0}}, 
                                                            Dims{4, {cv1_dim.d[0], cv1_dim.d[1]/2, cv1_dim.d[2], cv1_dim.d[3]}}, 
                                                            Dims{4, {1, 1, 1, 1}});
    s2->setName(std::string(lname + ".s2").data());                                                            
    
    aggregate_layers[0] = s1;
    aggregate_tensors[0] = aggregate_layers[0]->getOutput(0);
    aggregate_layers[1] = s2;
    aggregate_tensors[1] = aggregate_layers[1]->getOutput(0);
    last_layer_idx = 1;
    for(int i=0; i<n; i++){
        Dims s2_dim = aggregate_layers[last_layer_idx]->getOutput(0)->getDimensions();
        aggregate_layers[2+i] = bottleneck(network, weightMap, *aggregate_layers[last_layer_idx]->getOutput(0), s2_dim.d[1], c, shortcut, 1, 1.0, lname + ".m." + std::to_string(i));
        aggregate_tensors[2+i] = aggregate_layers[2+i]->getOutput(0);
        last_layer_idx = 2+i;
    }
    auto cat = network->addConcatenation(aggregate_tensors, (last_layer_idx+1));
    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), outch, 1, 1, 1, lname + ".cv2");

    return cv2;
}

ILayer *C2fCIB(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int n, bool shortcut, bool lk, int g, float e, std::string lname)
{
    ILayer* aggregate_layers[10];
    ITensor* aggregate_tensors[10];
    int last_layer_idx = 0;
    if(n>8){
        perror("Bottleneck counts axceed buffer, should not be more than 8\n");
        exit(0);
    }
    int c = int(outch*e);
    auto cv1 = convBlock(network, weightMap, input, c*2, 1, 1, 1, lname + ".cv1");
    Dims cv1_dim = cv1->getOutput(0)->getDimensions();

    ISliceLayer *s1 = network->addSlice(*cv1->getOutput(0), Dims{4, {0, 0, 0, 0}}, 
                                                            Dims{4, {cv1_dim.d[0], cv1_dim.d[1]/2, cv1_dim.d[2], cv1_dim.d[3]}}, 
                                                            Dims{4, {1, 1, 1, 1}});
    s1->setName(std::string(lname + ".s1").data());
    ISliceLayer *s2 = network->addSlice(*cv1->getOutput(0), Dims{4, {0, cv1_dim.d[1]/2, 0, 0}}, 
                                                            Dims{4, {cv1_dim.d[0], cv1_dim.d[1]/2, cv1_dim.d[2], cv1_dim.d[3]}}, 
                                                            Dims{4, {1, 1, 1, 1}});
    s2->setName(std::string(lname + ".s2").data());                                                            
    
    aggregate_layers[0] = s1;
    aggregate_tensors[0] = aggregate_layers[0]->getOutput(0);
    aggregate_layers[1] = s2;
    aggregate_tensors[1] = aggregate_layers[1]->getOutput(0);
    last_layer_idx = 1;
    for(int i=0; i<n; i++){
        Dims s2_dim = aggregate_layers[last_layer_idx]->getOutput(0)->getDimensions();
        aggregate_layers[2+i] = CIB(network, weightMap, *aggregate_layers[last_layer_idx]->getOutput(0), c, shortcut, 1, 1.0, false, lname + ".m." + std::to_string(i));
        aggregate_tensors[2+i] = aggregate_layers[2+i]->getOutput(0);
        last_layer_idx = 2+i;
    }
    auto cat = network->addConcatenation(aggregate_tensors, (last_layer_idx+1));
    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), outch, 1, 1, 1, lname + ".cv2");

    return cv2;
}



ILayer *focus(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int inch, int outch, int ksize, std::string lname)
{
    ISliceLayer *s1 = network->addSlice(input, Dims3{0, 0, 0}, Dims3{inch, yolov10_model_input_height / 2, yolov10_model_input_width / 2}, Dims3{1, 2, 2});
    ISliceLayer *s2 = network->addSlice(input, Dims3{0, 1, 0}, Dims3{inch, yolov10_model_input_height / 2, yolov10_model_input_width / 2}, Dims3{1, 2, 2});
    ISliceLayer *s3 = network->addSlice(input, Dims3{0, 0, 1}, Dims3{inch, yolov10_model_input_height / 2, yolov10_model_input_width / 2}, Dims3{1, 2, 2});
    ISliceLayer *s4 = network->addSlice(input, Dims3{0, 1, 1}, Dims3{inch, yolov10_model_input_height / 2, yolov10_model_input_width / 2}, Dims3{1, 2, 2});
    ITensor *inputTensors[] = {s1->getOutput(0), s2->getOutput(0), s3->getOutput(0), s4->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto conv = convBlock(network, weightMap, *cat->getOutput(0), outch, ksize, 1, 1, lname + ".conv");
    return conv;
}

ILayer *bottleneck(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, bool shortcut, int g, float e, std::string lname)
{
    auto cv1 = convBlock(network, weightMap, input, (int)((float)c2 * e), 3, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, 3, 1, g, lname + ".cv2");
    if (shortcut && c1 == c2){
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

ILayer *CIB(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c2, bool shortcut, int g, float e, bool lk, std::string lname)
{
    Dims input_dims = input.getDimensions();
    int c1 = input_dims.d[1];
    int c_ = (int)((float)c2 * e);

    auto cv1 = convBlock(network, weightMap, input, c1, 3, 1, c1, lname + ".cv1.0");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), 2*c_, 1, 1, 1, lname + ".cv1.1");
    auto cv3 = convBlock(network, weightMap, *cv2->getOutput(0), 2*c_, 3, 1, 2*c_, lname + ".cv1.2");
    auto cv4 = convBlock(network, weightMap, *cv3->getOutput(0), c2, 1, 1, 1, lname + ".cv1.3");
    auto cv5 = convBlock(network, weightMap, *cv4->getOutput(0), c2, 3, 1, c2, lname + ".cv1.4");
    if (shortcut && c1 == c2){
        auto ew = network->addElementWise(input, *cv5->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv5;
}

ILayer *SCDown(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c2, int k, int s, std::string lname)
{
    auto cv1 = convBlock(network, weightMap, input, c2, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), c2, k, s, c2, lname + ".cv2", false);
    return cv2;
}

ILayer *bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = network->addConvolutionNd(input, c_, DimsHW{1, 1}, weightMap[lname + ".cv2.weight"], emptywts);
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++)
    {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }
    auto cv3 = network->addConvolutionNd(*y1, c_, DimsHW{1, 1}, weightMap[lname + ".cv3.weight"], emptywts);

    ITensor *inputTensors[] = {cv3->getOutput(0), cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    IScaleLayer *bn = addBatchNorm2d(network, weightMap, *cat->getOutput(0), lname + ".bn", 1e-4);
    auto lr = network->addActivation(*bn->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    auto cv4 = convBlock(network, weightMap, *lr->getOutput(0), c2, 1, 1, 1, lname + ".cv4");
    return cv4;
}

ILayer *C3(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname)
{
    int c_ = (int)((float)c2 * e);
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv2");
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++)
    {
        auto b = bottleneck(network, weightMap, *y1, c_, c_, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }

    ITensor *inputTensors[] = {y1, cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv3");
    return cv3;
}

ILayer *SPP(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int k1, int k2, int k3, std::string lname)
{
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k1, k1});
    pool1->setPaddingNd(DimsHW{k1 / 2, k1 / 2});
    pool1->setStrideNd(DimsHW{1, 1});
    auto pool2 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k2, k2});
    pool2->setPaddingNd(DimsHW{k2 / 2, k2 / 2});
    pool2->setStrideNd(DimsHW{1, 1});
    auto pool3 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k3, k3});
    pool3->setPaddingNd(DimsHW{k3 / 2, k3 / 2});
    pool3->setStrideNd(DimsHW{1, 1});

    ITensor *inputTensors[] = {cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);

    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

ILayer *SPPF(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, int k, std::string lname)
{
    int c_ = c1 / 2;
    auto cv1 = convBlock(network, weightMap, input, c_, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k, k});
    pool1->setPaddingNd(DimsHW{k / 2, k / 2});
    pool1->setStrideNd(DimsHW{1, 1});
    auto pool2 = network->addPoolingNd(*pool1->getOutput(0), PoolingType::kMAX, DimsHW{k, k});
    pool2->setPaddingNd(DimsHW{k / 2, k / 2});
    pool2->setStrideNd(DimsHW{1, 1});
    auto pool3 = network->addPoolingNd(*pool2->getOutput(0), PoolingType::kMAX, DimsHW{k, k});
    pool3->setPaddingNd(DimsHW{k / 2, k / 2});
    pool3->setStrideNd(DimsHW{1, 1});
    ITensor *inputTensors[] = {cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    Dims cat_dim = cat->getOutput(0)->getDimensions();
    // printf("In SPPF, cat shape(%d, %d, %d)\n", cat_dim.d[0], cat_dim.d[1], cat_dim.d[2]);
    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), c2, 1, 1, 1, lname + ".cv2");
    return cv2;
}

ILayer *Attention(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int dim, int num_heads, float attn_ratio, std::string lname)
{
    int head_dim = static_cast<int>(static_cast<float>(dim) / static_cast<float>(num_heads));
    int key_dim = static_cast<int>(static_cast<float>(head_dim) * attn_ratio);
    float scale = std::pow(static_cast<float>(key_dim), -0.5);
    int nh_kd = key_dim * num_heads;
    int h = dim + nh_kd * 2;

    // printf("head_dim: %d\n", head_dim);
    // printf("key_dim: %d\n", key_dim);
    // printf("scale: %f\n", scale);
    // printf("nh_kd: %d\n", nh_kd);
    // printf("h: %d\n", h);

    Dims input_dims = input.getDimensions(); ///< 输入shape

    /** QKV */
    auto qkv = convBlock(network, weightMap, input, h, 1, 1, 1, lname + ".qkv", false);
    /** Reshape. */
    // Dims qkv_dims = qkv->getOutput(0)->getDimensions();
    // for(int i=0; i<qkv_dims.nbDims; i++){
    //     printf("qkv_dims.d[%d]: %d\n", i, qkv_dims.d[i]);
    // }
    Dims reshape_dims;
    reshape_dims.nbDims = 4;
    reshape_dims.d[0] = input_dims.d[0];
    reshape_dims.d[1] = num_heads;
    reshape_dims.d[2] = key_dim*2+head_dim;
    reshape_dims.d[3] = input_dims.d[2] * input_dims.d[3];
    
    nvinfer1::IShuffleLayer *qkv_reshape = network->addShuffle(*qkv->getOutput(0));
    qkv_reshape->setReshapeDimensions(reshape_dims);

    /** Split. */
    Dims qkv_reshape_dim = qkv_reshape->getOutput(0)->getDimensions();
    ISliceLayer *q = network->addSlice(*qkv_reshape->getOutput(0), Dims{4, {0, 0, 0, 0}}, 
                                                           Dims{4, {qkv_reshape_dim.d[0], qkv_reshape_dim.d[1], key_dim, qkv_reshape_dim.d[3]}}, 
                                                           Dims{4, {1, 1, 1, 1}});
    q->setName(std::string(lname + ".q").data());

    ISliceLayer *k = network->addSlice(*qkv_reshape->getOutput(0), Dims{4, {0, 0, key_dim, 0}}, 
                                                           Dims{4, {qkv_reshape_dim.d[0], qkv_reshape_dim.d[1], key_dim, qkv_reshape_dim.d[3]}}, 
                                                           Dims{4, {1, 1, 1, 1}});
    k->setName(std::string(lname + ".k").data());

    ISliceLayer *v = network->addSlice(*qkv_reshape->getOutput(0), Dims{4, {0, 0, key_dim*2, 0}}, 
                                                           Dims{4, {qkv_reshape_dim.d[0], qkv_reshape_dim.d[1], head_dim, qkv_reshape_dim.d[3]}}, 
                                                           Dims{4, {1, 1, 1, 1}});
    v->setName(std::string(lname + ".v").data());

    // Dims q_dim = q->getOutput(0)->getDimensions();
    // for(int i=0; i<q_dim.nbDims; i++){
    //     printf("q_dim.d[%d]: %d\n", i, q_dim.d[i]);
    // }

#if 0
    /** qt = q.transpose */
    auto q_transpose = network->addShuffle(*q->getOutput(0));
    q_transpose->setFirstTranspose(Permutation{0, 1, 3, 2}); ///< shape(1, 2, M)
    Dims q_transpose_dim = q_transpose->getOutput(0)->getDimensions();
    for(int i=0; i<q_transpose_dim.nbDims; i++){
        printf("q_transpose_dim.d[%d]: %d\n", i, q_transpose_dim.d[i]);
    }
    /** qt * k */
    // TODO
#else
    /** qtk = q.transpose * k */
    auto qtk = network->addMatrixMultiply(*q->getOutput(0), MatrixOperation::kTRANSPOSE, *k->getOutput(0), MatrixOperation::kNONE);
    // Dims qtk_dim = qtk->getOutput(0)->getDimensions();
    // for(int i=0; i<qtk_dim.nbDims; i++){
    //     printf("qtk_dim.d[%d]: %d\n", i, qtk_dim.d[i]);
    // }
    /** qtk * scale */
    Weights tensor_scale;
    tensor_scale.type   = DataType::kFLOAT; ///< or kHALF
    tensor_scale.count  = 1;
    tensor_scale.values = new float[1]{scale};

    float shift = 0.0;
    Weights tensor_shift;
    tensor_shift.type   = DataType::kFLOAT; ///< or kHALF
    tensor_shift.count  = 1;
    tensor_shift.values = new float[1]{shift};

    float power = 1.0;
    Weights tensor_power;
    tensor_power.type   = DataType::kFLOAT; ///< or kHALF
    tensor_power.count  = 1;
    tensor_power.values = new float[1]{power};

    // Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto attn = network->addScale(*qtk->getOutput(0), ScaleMode::kUNIFORM, tensor_shift, tensor_scale, tensor_power); ///< shift, scale, power
#endif

    /** softmax */
    ISoftMaxLayer* attn_softmax = network->addSoftMax(*attn->getOutput(0));
    attn_softmax->setAxes(0x08); ///< 对最后一个维度执行softmax

    /** va = v * attn_softmax.T */
    auto va = network->addMatrixMultiply(*v->getOutput(0), MatrixOperation::kNONE, *attn_softmax->getOutput(0), MatrixOperation::kTRANSPOSE);
    // Dims va_dim = va->getOutput(0)->getDimensions();
    // for(int i=0; i<va_dim.nbDims; i++){
    //     printf("va_dim.d[%d]: %d\n", i, va_dim.d[i]);
    // }
    
    /** reshape(va) to shape(NCHW) */
    nvinfer1::IShuffleLayer *va_reshape = network->addShuffle(*va->getOutput(0));
    va_reshape->setReshapeDimensions(Dims{4, {input_dims.d[0], input_dims.d[1], input_dims.d[2], input_dims.d[3]}});

    /** v.reshape */
    nvinfer1::IShuffleLayer *v_reshape = network->addShuffle(*v->getOutput(0));
    v_reshape->setReshapeDimensions(Dims{4, {input_dims.d[0], input_dims.d[1], input_dims.d[2], input_dims.d[3]}});
    /** pe(v) */
    auto pe = convBlock(network, weightMap, *v_reshape->getOutput(0), dim, 3, 1, dim, lname + ".pe", false);

    auto x = network->addElementWise(*va_reshape->getOutput(0), *pe->getOutput(0), ElementWiseOperation::kSUM);

    auto proj = convBlock(network, weightMap, *x->getOutput(0), dim, 1, 1, 1, lname + ".proj", false);

    return proj;
}

ILayer *PSA(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c1, int c2, float e, std::string lname)
{
    int c = static_cast<int>(static_cast<float>(c1) * 0.5);
    auto cv1 = convBlock(network, weightMap, input, 2*c, 1, 1, 1, lname + ".cv1");

    /** 对cv1的输出在通道轴上做split操作。
     */
    Dims cv1_dim = cv1->getOutput(0)->getDimensions();
    ISliceLayer *a = network->addSlice(*cv1->getOutput(0), Dims{4, {0, 0, 0, 0}}, 
                                                            Dims{4, {cv1_dim.d[0], cv1_dim.d[1]/2, cv1_dim.d[2], cv1_dim.d[3]}}, 
                                                            Dims{4, {1, 1, 1, 1}});
    a->setName(std::string(lname + ".a").data());

    ISliceLayer *b = network->addSlice(*cv1->getOutput(0), Dims{4, {0, cv1_dim.d[1]/2, 0, 0}}, 
                                                            Dims{4, {cv1_dim.d[0], cv1_dim.d[1]/2, cv1_dim.d[2], cv1_dim.d[3]}}, 
                                                            Dims{4, {1, 1, 1, 1}});
    b->setName(std::string(lname + ".b").data());

    /** attn(b) */
    auto attn_b = Attention(network, weightMap, *b->getOutput(0), c, int(c/64), 0.5, lname+".attn");

    /** b + attn(b) */
    auto b_attn_b = network->addElementWise(*b->getOutput(0), *attn_b->getOutput(0), ElementWiseOperation::kSUM);

    /** ffn(b_attn_b) */
    auto ffn_0 = convBlock(network, weightMap, *b_attn_b->getOutput(0), 2*c, 1, 1, 1, lname + ".ffn.0");
    auto ffn_1 = convBlock(network, weightMap, *ffn_0->getOutput(0), c, 1, 1, 1, lname + ".ffn.1", false); ///< ffn.1不使用激活函数

    /** b_attn_b + ffn(b_attn_b) */
    auto b_ffn_b = network->addElementWise(*b_attn_b->getOutput(0), *ffn_1->getOutput(0), ElementWiseOperation::kSUM);

    /** cat(a, b), 在C通道执行Concatenation. */
    ITensor *ab_cat_inputTensors[] = {a->getOutput(0), b_ffn_b->getOutput(0)};
    auto ab_cat = network->addConcatenation(ab_cat_inputTensors, 2);

    auto psa_out = convBlock(network, weightMap, *ab_cat->getOutput(0), c1, 1, 1, 1, lname + ".cv2");

    return psa_out;
}


IPluginV2Layer *addDetectHead(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, std::string lname, std::vector<IConvolutionLayer *> dets)
{
    auto creator = getPluginRegistry()->getPluginCreator("YOLOv5_YoloLayerPlugin", "1.0");
    auto anchors = getAnchors(weightMap, lname);
    PluginField plugin_fields[2];
    int netinfo[4] = {yolov10_num_classes, yolov10_model_input_width, yolov10_model_input_height, MAX_OUTPUT_BBOX_COUNT};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kINT32;
    int scale = 8;
    std::vector<YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++){
        // printf("%f, %f, %f, %f, %f, %f\n", anchors[i][0], anchors[i][1], anchors[i][2], anchors[i][3], anchors[i][4], anchors[i][5]);
        YoloKernel kernel;
        kernel.width = yolov10_model_input_width / scale;
        kernel.height = yolov10_model_input_height / scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<ITensor *> input_tensors;
    for (auto det : dets)
    {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
}


} ///< NAMESPACE_YOLOv10
