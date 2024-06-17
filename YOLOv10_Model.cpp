#include "YOLOv10_Model.h"


/**
 * @brief Get parameters from config file.
 */
int yolov8_model_input_width = -1;
int yolov8_model_input_height = -1;
int yolov8_num_classes = -1;
float yolov8_nms_thresh = 0.0;
float yolov8_bbox_conf_thresh = 0.0;
std::string yolov8_sub_edition;



YOLOv10_Model::YOLOv10_Model(): Model()
{
    batchsize = 1;
    inputIndex = -1;
    scores_output_index = -1;
    classes_output_index = -1;
    boxes_output_index = -1;
    /** Parameters parser */
    ParseParams();
    
    /** 输入大小 */
    mModelInputDataSize = batchsize * 3 * yolov8_model_input_height * yolov8_model_input_width;
    
    /** 输出大小 */
    mScoresOutputSize  = batchsize * NAMESPACE_YOLOv10::MAX_OUTPUT_BBOX_COUNT * 1; ///<
    mClassesOutputSize = batchsize * NAMESPACE_YOLOv10::MAX_OUTPUT_BBOX_COUNT * 1; ///<
    mBoxesOutputSize   = batchsize * NAMESPACE_YOLOv10::MAX_OUTPUT_BBOX_COUNT * 4; ///<

    /** Output buffer name */
    mInputBlobName = "data";
    mBoxesBlobName    = "BoxesTensor";
    mScoresBlobName   = "ScoresTensor";
    mClassesBlobName  = "ClassesTensor";

    mInputDims.nbDims = 4;
    mInputDims.d[0] = batchsize;
    mInputDims.d[1] = 3;
    mInputDims.d[2] = yolov8_model_input_height;
    mInputDims.d[3] = yolov8_model_input_width;

    reg_max = 16;

    /** Preset anchor */
    for(int h=0; h<FM0_H; h++){
        for(int w=0; w<FM0_W; w++){
            fm0_anchor_grid[h*80*2 + w*2 + 0] = w*1.0+0.5;
            fm0_anchor_grid[h*80*2 + w*2 + 1] = h*1.0+0.5;
        }
    }
    for(int h=0; h<FM1_H; h++){
        for(int w=0; w<FM1_W; w++){
            fm1_anchor_grid[h*40*2 + w*2 + 0] = w*1.0+0.5;
            fm1_anchor_grid[h*40*2 + w*2 + 1] = h*1.0+0.5;
        }
    }
    for(int h=0; h<FM2_H; h++){
        for(int w=0; w<FM2_W; w++){
            fm2_anchor_grid[h*20*2 + w*2 + 0] = w*1.0+0.5;
            fm2_anchor_grid[h*20*2 + w*2 + 1] = h*1.0+0.5;
        }
    }

    /** Preset stride */
    for(int i=0; i<(FM0_H*FM0_W+FM1_H*FM1_W+FM2_H*FM2_W); i++){
        if(i<FM0_H*FM0_W){
            stride[i] = 8.0;
        }else if(i>=FM0_H*FM0_W && i<(FM0_H*FM0_W+FM1_H*FM1_W)){
            stride[i] = 16.0;
        }else{
            stride[i] = 32.0;
        }   
    }
}

YOLOv10_Model::~YOLOv10_Model()
{
}

int YOLOv10_Model::get_width(int x, float gw, int divisor)
{
    /** [SAI-KEY] YOLOv10中有对通道的最大限制. */
    if(x > 768){
        x = 768;
    }
    return int(ceil((x * gw) / divisor)) * divisor;
}

int YOLOv10_Model::get_depth(int x, float gd)
{
    if (x == 1)
        return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0)
    {
        --r;
    }
    return std::max<int>(r, 1);
}


/**
 * @brief Create YOLOv10 engine
 * 
 * @param maxBatchSize 
 * @param builder 
 * @param config 
 * @param dt 
 * @param wts_file 
 * @param gd 
 * @param gw 
 * @param compute_mode 
 * @return ICudaEngine* 
 */
ICudaEngine* YOLOv10_Model::BuildEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, 
                                            std::string wts_file, float &gd, float &gw, ComputationMode compute_mode) 
{
    INetworkDefinition *network = builder->createNetworkV2(1U);

    ITensor *data = network->addInput(mInputBlobName, dt, mInputDims);
    // assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_file);
    /** yolov8 backbone */
    auto conv_0 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *data, get_width(64, gw), 3, 2, 1, "model.0");
    auto conv_1 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *conv_0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto  c2f_2 = NAMESPACE_YOLOv10::C2f(network, weightMap, *conv_1->getOutput(0), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv_3 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *c2f_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto  c2f_4 = NAMESPACE_YOLOv10::C2f(network, weightMap, *conv_3->getOutput(0), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");

    auto scdown_5 = NAMESPACE_YOLOv10::SCDown(network, weightMap, *c2f_4->getOutput(0), get_width(512, gw), 3, 2, "model.5");

    auto  c2f_6 = NAMESPACE_YOLOv10::C2f(network, weightMap, *scdown_5->getOutput(0), get_width(512, gw), get_depth(6, gd), true, 1, 0.5, "model.6");

    auto scdown_7 = NAMESPACE_YOLOv10::SCDown(network, weightMap, *c2f_6->getOutput(0), get_width(1024, gw), 3, 2, "model.7"); ///< [SAI-KEY] 此处的1024会被限制为768

    auto c2fcib_8 = NAMESPACE_YOLOv10::C2fCIB(network, weightMap, *scdown_7->getOutput(0), get_width(1024, gw), get_depth(3, gd), true, false, 1, 0.5, "model.8");

    Dims c2fcib_8_dim = c2fcib_8->getOutput(0)->getDimensions();
    auto  sppf_9 = NAMESPACE_YOLOv10::SPPF(network, weightMap, *c2fcib_8->getOutput(0), c2fcib_8_dim.d[1], get_width(1024, gw), 5, "model.9");

    Dims sppf9_dim = sppf_9->getOutput(0)->getDimensions();
    auto psa_10 = NAMESPACE_YOLOv10::PSA(network, weightMap, *sppf_9->getOutput(0), sppf9_dim.d[1], get_width(1024, gw), 0.5, "model.10");

    Dims psa_10_dim = psa_10->getOutput(0)->getDimensions();
    auto upsample_11 = network->addResize(*psa_10->getOutput(0));
    upsample_11->setResizeMode(ResizeMode::kNEAREST);
    upsample_11->setOutputDimensions(Dims{4, {psa_10_dim.d[0], psa_10_dim.d[1], (psa_10_dim.d[2] * 2), (psa_10_dim.d[3] * 2)}});

    /** M12 */
    ITensor* inputTensors12[] = {upsample_11->getOutput(0), c2f_6->getOutput(0)};
    auto cat_12 = network->addConcatenation(inputTensors12, 2);
    
    /** M13 */
    auto  c2f_13 = NAMESPACE_YOLOv10::C2f(network, weightMap, *cat_12->getOutput(0), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");

    /** M14 */
    Dims c2f_13_dim = c2f_13->getOutput(0)->getDimensions();
    auto upsample_14 = network->addResize(*c2f_13->getOutput(0));
    upsample_14->setResizeMode(ResizeMode::kNEAREST);
    upsample_14->setOutputDimensions(Dims{4, {c2f_13_dim.d[0], c2f_13_dim.d[1], (c2f_13_dim.d[2] * 2), (c2f_13_dim.d[3] * 2)}});
    
    /** M15 */
    ITensor* inputTensors15[] = {upsample_14->getOutput(0), c2f_4->getOutput(0)};
    auto cat_15 = network->addConcatenation(inputTensors15, 2);
    
    /** M16 */
    auto  c2f_16 = NAMESPACE_YOLOv10::C2f(network, weightMap, *cat_15->getOutput(0), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.16");  ///< to DetectHead

    /** M17 */
    auto conv_17 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *c2f_16->getOutput(0), get_width(256, gw), 3, 2, 1, "model.17");

    /** M18 */
    ITensor* inputTensors18[] = {conv_17->getOutput(0), c2f_13->getOutput(0)};
    auto cat_18 = network->addConcatenation(inputTensors18, 2);

    /** M19 */
    auto c2fcib_19 = NAMESPACE_YOLOv10::C2fCIB(network, weightMap, *cat_18->getOutput(0), get_width(512, gw), get_depth(3, gd), true, false, 1, 0.5, "model.19");

    /** M20 */
    auto scdown_20 = NAMESPACE_YOLOv10::SCDown(network, weightMap, *c2fcib_19->getOutput(0), get_width(512, gw), 3, 2, "model.20");

    /** M21 */
    ITensor* inputTensors21[] = {scdown_20->getOutput(0), psa_10->getOutput(0)};
    auto cat_21 = network->addConcatenation(inputTensors21, 2);

    /** M22 */
    auto c2fcib_22 = NAMESPACE_YOLOv10::C2fCIB(network, weightMap, *cat_21->getOutput(0), get_width(1024, gw), get_depth(3, gd), true, false, 1, 0.5, "model.22");

    /** 
     * forward_feat
     */
    // 首先以c2f_16的输出通道为基准计算c2和c3, c2和c3分别是forward_feat中的cv2和cv3的中间通道数。
    Dims c2f_16_dim = c2f_16->getOutput(0)->getDimensions();
    Dims c2fcib_19_dim = c2fcib_19->getOutput(0)->getDimensions();
    Dims c2fcib_22_dim = c2fcib_22->getOutput(0)->getDimensions();
    int c2f_16_channel = c2f_16_dim.d[1];
    int c2_app = static_cast<int>(static_cast<float>(c2f_16_channel)/static_cast<float>(4));
    int c2 = std::max(16, std::max(c2_app, 16*4));
    // printf("[forward_feat] c2: %d\n", c2);
    int c3 = std::max(c2f_16_channel, std::min(yolov8_num_classes, 100));
    // printf("[forward_feat] c3: %d\n", c3);
    
    /** c2f_16(x0): one2one_cv2 */
    auto forward_feat_cv2_x0_0 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *c2f_16->getOutput(0), c2, 3, 1, 1, "model.23.one2one_cv2.0.0");
    auto forward_feat_cv2_x0_1 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv2_x0_0->getOutput(0), c2, 3, 1, 1, "model.23.one2one_cv2.0.1");
    IConvolutionLayer *forward_feat_cv2_x0 = network->addConvolutionNd(*forward_feat_cv2_x0_1->getOutput(0), 4*reg_max, DimsHW{1, 1}, weightMap["model.23.one2one_cv2.0.2.weight"], weightMap["model.23.one2one_cv2.0.2.bias"]); ///< cv2输出
    forward_feat_cv2_x0->setName(std::string("model.23.forward_feat.one2one_cv2.x0").data());
    forward_feat_cv2_x0->setStrideNd(DimsHW{1, 1}); ///< 默认就是DimsHW{1, 1}, 此处可以不添加
    forward_feat_cv2_x0->setPaddingNd(DimsHW{0, 0});
    forward_feat_cv2_x0->setNbGroups(1);
    /** c2f_16(x0): one2one_cv3 */
    auto forward_feat_cv3_x0_0 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *c2f_16->getOutput(0), c2f_16_dim.d[1], 3, 1, c2f_16_dim.d[1], "model.23.one2one_cv3.0.0.0");
    auto forward_feat_cv3_x0_1 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv3_x0_0->getOutput(0), c3, 1, 1, 1, "model.23.one2one_cv3.0.0.1");
    auto forward_feat_cv3_x0_2 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv3_x0_1->getOutput(0), c3, 3, 1, c3, "model.23.one2one_cv3.0.1.0");
    auto forward_feat_cv3_x0_3 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv3_x0_2->getOutput(0), c3, 1, 1, 1, "model.23.one2one_cv3.0.1.1");
    IConvolutionLayer *forward_feat_cv3_x0 = network->addConvolutionNd(*forward_feat_cv3_x0_3->getOutput(0), yolov8_num_classes, DimsHW{1, 1}, weightMap["model.23.one2one_cv3.0.2.weight"], weightMap["model.23.one2one_cv3.0.2.bias"]); ///< cv2输出
    forward_feat_cv3_x0->setName(std::string("model.23.forward_feat.one2one_cv3.x0").data());
    forward_feat_cv3_x0->setStrideNd(DimsHW{1, 1}); ///< 默认就是DimsHW{1, 1}, 此处可以不添加
    forward_feat_cv3_x0->setPaddingNd(DimsHW{0, 0});
    forward_feat_cv3_x0->setNbGroups(1);

#if 0
    /** M23, forward_feat, y0 */
    ITensor* M23_forward_feat_y0[] = {forward_feat_cv2_x0->getOutput(0), forward_feat_cv3_x0->getOutput(0)};
    auto y0 = network->addConcatenation(M23_forward_feat_y0, 2);
#else
    /** 为了推理方便，直接将box和cls分支先做reshape */
    Dims box_0_dims = forward_feat_cv2_x0->getOutput(0)->getDimensions();
    Dims box_0_reshape_dims;
    box_0_reshape_dims.nbDims = 3;
    box_0_reshape_dims.d[0] = box_0_dims.d[0];
    box_0_reshape_dims.d[1] = box_0_dims.d[1];
    box_0_reshape_dims.d[2] = box_0_dims.d[2] * box_0_dims.d[3];
    nvinfer1::IShuffleLayer *box_0 = network->addShuffle(*forward_feat_cv2_x0->getOutput(0)); ///< shape(144, 6400)
    box_0->setReshapeDimensions(box_0_reshape_dims);

    Dims cls_0_dims = forward_feat_cv3_x0->getOutput(0)->getDimensions();
    Dims cls_0_reshape_dims;
    cls_0_reshape_dims.nbDims = 3;
    cls_0_reshape_dims.d[0] = cls_0_dims.d[0];
    cls_0_reshape_dims.d[1] = cls_0_dims.d[1];
    cls_0_reshape_dims.d[2] = cls_0_dims.d[2] * cls_0_dims.d[3];
    nvinfer1::IShuffleLayer *cls_0 = network->addShuffle(*forward_feat_cv3_x0->getOutput(0)); ///< shape(144, 6400)
    cls_0->setReshapeDimensions(cls_0_reshape_dims);
#endif

    /** c2fcib_19(x1): one2one_cv2 */
    auto forward_feat_cv2_x1_0 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *c2fcib_19->getOutput(0), c2, 3, 1, 1, "model.23.one2one_cv2.1.0");
    auto forward_feat_cv2_x1_1 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv2_x1_0->getOutput(0), c2, 3, 1, 1, "model.23.one2one_cv2.1.1");
    IConvolutionLayer *forward_feat_cv2_x1 = network->addConvolutionNd(*forward_feat_cv2_x1_1->getOutput(0), 4*reg_max, DimsHW{1, 1}, weightMap["model.23.one2one_cv2.1.2.weight"], weightMap["model.23.one2one_cv2.1.2.bias"]); ///< cv2输出
    forward_feat_cv2_x1->setName(std::string("model.23.forward_feat.one2one_cv2.x1").data());
    forward_feat_cv2_x1->setStrideNd(DimsHW{1, 1}); ///< 默认就是DimsHW{1, 1}, 此处可以不添加
    forward_feat_cv2_x1->setPaddingNd(DimsHW{0, 0});
    forward_feat_cv2_x1->setNbGroups(1);
    /** c2fcib_19(x1): one2one_cv3 */
    auto forward_feat_cv3_x1_0 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *c2fcib_19->getOutput(0), c2fcib_19_dim.d[1], 3, 1, c2fcib_19_dim.d[1], "model.23.one2one_cv3.1.0.0");
    auto forward_feat_cv3_x1_1 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv3_x1_0->getOutput(0), c3, 1, 1, 1, "model.23.one2one_cv3.1.0.1");
    auto forward_feat_cv3_x1_2 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv3_x1_1->getOutput(0), c3, 3, 1, c3, "model.23.one2one_cv3.1.1.0");
    auto forward_feat_cv3_x1_3 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv3_x1_2->getOutput(0), c3, 1, 1, 1, "model.23.one2one_cv3.1.1.1");
    IConvolutionLayer *forward_feat_cv3_x1 = network->addConvolutionNd(*forward_feat_cv3_x1_3->getOutput(0), yolov8_num_classes, DimsHW{1, 1}, weightMap["model.23.one2one_cv3.1.2.weight"], weightMap["model.23.one2one_cv3.1.2.bias"]); ///< cv2输出
    forward_feat_cv3_x1->setName(std::string("model.23.forward_feat.one2one_cv3.x1").data());
    forward_feat_cv3_x1->setStrideNd(DimsHW{1, 1}); ///< 默认就是DimsHW{1, 1}, 此处可以不添加
    forward_feat_cv3_x1->setPaddingNd(DimsHW{0, 0});
    forward_feat_cv3_x1->setNbGroups(1);

#if 0
    /** M23, forward_feat, y1 */
    ITensor* M23_forward_feat_y1[] = {forward_feat_cv2_x1->getOutput(0), forward_feat_cv3_x1->getOutput(0)};
    auto y1 = network->addConcatenation(M23_forward_feat_y1, 2);
#else
    /** 为了推理方便，直接将box和cls分支先做reshape */
    Dims box_1_dims = forward_feat_cv2_x1->getOutput(0)->getDimensions();
    Dims box_1_reshape_dims;
    box_1_reshape_dims.nbDims = 3;
    box_1_reshape_dims.d[0] = box_1_dims.d[0];
    box_1_reshape_dims.d[1] = box_1_dims.d[1];
    box_1_reshape_dims.d[2] = box_1_dims.d[2] * box_1_dims.d[3];
    nvinfer1::IShuffleLayer *box_1 = network->addShuffle(*forward_feat_cv2_x1->getOutput(0)); ///< shape(144, 6400)
    box_1->setReshapeDimensions(box_1_reshape_dims);

    Dims cls_1_dims = forward_feat_cv3_x1->getOutput(0)->getDimensions();
    Dims cls_1_reshape_dims;
    cls_1_reshape_dims.nbDims = 3;
    cls_1_reshape_dims.d[0] = cls_1_dims.d[0];
    cls_1_reshape_dims.d[1] = cls_1_dims.d[1];
    cls_1_reshape_dims.d[2] = cls_1_dims.d[2] * cls_1_dims.d[3];
    nvinfer1::IShuffleLayer *cls_1 = network->addShuffle(*forward_feat_cv3_x1->getOutput(0)); ///< shape(144, 6400)
    cls_1->setReshapeDimensions(cls_1_reshape_dims);
#endif

    /** c2fcib_22(x2): one2one_cv2 */
    auto forward_feat_cv2_x2_0 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *c2fcib_22->getOutput(0), c2, 3, 1, 1, "model.23.one2one_cv2.2.0");
    auto forward_feat_cv2_x2_1 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv2_x2_0->getOutput(0), c2, 3, 1, 1, "model.23.one2one_cv2.2.1");
    IConvolutionLayer *forward_feat_cv2_x2 = network->addConvolutionNd(*forward_feat_cv2_x2_1->getOutput(0), 4*reg_max, DimsHW{1, 1}, weightMap["model.23.one2one_cv2.2.2.weight"], weightMap["model.23.one2one_cv2.2.2.bias"]); ///< cv2输出
    forward_feat_cv2_x2->setName(std::string("model.23.forward_feat.one2one_cv2.x2").data());
    forward_feat_cv2_x2->setStrideNd(DimsHW{1, 1}); ///< 默认就是DimsHW{1, 1}, 此处可以不添加
    forward_feat_cv2_x2->setPaddingNd(DimsHW{0, 0});
    forward_feat_cv2_x2->setNbGroups(1);
    /** c2fcib_22(x2): one2one_cv3 */
    auto forward_feat_cv3_x2_0 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *c2fcib_22->getOutput(0), c2fcib_22_dim.d[1], 3, 1, c2fcib_22_dim.d[1], "model.23.one2one_cv3.2.0.0");
    auto forward_feat_cv3_x2_1 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv3_x2_0->getOutput(0), c3, 1, 1, 1, "model.23.one2one_cv3.2.0.1");
    auto forward_feat_cv3_x2_2 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv3_x2_1->getOutput(0), c3, 3, 1, c3, "model.23.one2one_cv3.2.1.0");
    auto forward_feat_cv3_x2_3 = NAMESPACE_YOLOv10::convBlock(network, weightMap, *forward_feat_cv3_x2_2->getOutput(0), c3, 1, 1, 1, "model.23.one2one_cv3.2.1.1");
    IConvolutionLayer *forward_feat_cv3_x2 = network->addConvolutionNd(*forward_feat_cv3_x2_3->getOutput(0), yolov8_num_classes, DimsHW{1, 1}, weightMap["model.23.one2one_cv3.2.2.weight"], weightMap["model.23.one2one_cv3.2.2.bias"]); ///< cv2输出
    forward_feat_cv3_x2->setName(std::string("model.23.forward_feat.one2one_cv3.x2").data());
    forward_feat_cv3_x2->setStrideNd(DimsHW{1, 1}); ///< 默认就是DimsHW{1, 1}, 此处可以不添加
    forward_feat_cv3_x2->setPaddingNd(DimsHW{0, 0});
    forward_feat_cv3_x2->setNbGroups(1);

#if 0
    /** M23, forward_feat, y1 */
    ITensor* M23_forward_feat_y2[] = {forward_feat_cv2_x2->getOutput(0), forward_feat_cv3_x2->getOutput(0)};
    auto y2 = network->addConcatenation(M23_forward_feat_y2, 2);
#else
    /** 为了推理方便，直接将box和cls分支先做reshape */
    Dims box_2_dims = forward_feat_cv2_x2->getOutput(0)->getDimensions();
    Dims box_2_reshape_dims;
    box_2_reshape_dims.nbDims = 3;
    box_2_reshape_dims.d[0] = box_2_dims.d[0];
    box_2_reshape_dims.d[1] = box_2_dims.d[1];
    box_2_reshape_dims.d[2] = box_2_dims.d[2] * box_2_dims.d[3];
    nvinfer1::IShuffleLayer *box_2 = network->addShuffle(*forward_feat_cv2_x2->getOutput(0)); ///< shape(144, 6400)
    box_2->setReshapeDimensions(box_2_reshape_dims);

    Dims cls_2_dims = forward_feat_cv3_x2->getOutput(0)->getDimensions();
    Dims cls_2_reshape_dims;
    cls_2_reshape_dims.nbDims = 3;
    cls_2_reshape_dims.d[0] = cls_2_dims.d[0];
    cls_2_reshape_dims.d[1] = cls_2_dims.d[1];
    cls_2_reshape_dims.d[2] = cls_2_dims.d[2] * cls_2_dims.d[3];
    nvinfer1::IShuffleLayer *cls_2 = network->addShuffle(*forward_feat_cv3_x2->getOutput(0)); ///< shape(144, 6400)
    cls_2->setReshapeDimensions(cls_2_reshape_dims);
#endif

    /**  
     * [SAI-KEY] Inference
    */
    /** regression_box */
    ITensor* box_cat_tensors[] = {box_0->getOutput(0), box_1->getOutput(0), box_2->getOutput(0)};
    auto regression_box = network->addConcatenation(box_cat_tensors, 3);
    regression_box->setAxis(2);

    /** cls */
    ITensor* cls_cat_tensors[] = {cls_0->getOutput(0), cls_1->getOutput(0), cls_2->getOutput(0)};
    auto cls = network->addConcatenation(cls_cat_tensors, 3);
    cls->setAxis(2);

    // Dims cls_dims = cls->getOutput(0)->getDimensions();
    // printf("cls_dims.ndims: %d \n", cls_dims.nbDims);
    // for(int i=0; i<cls_dims.nbDims; i++){
    //     printf("dim[%d]: %d \n", i, cls_dims.d[i]);
    // }

    /** DFL(regression_box) */
    // if(reg_max > 1){
        Dims regression_box_dims = regression_box->getOutput(0)->getDimensions();
        Dims boxes_reshape_dims;
        boxes_reshape_dims.nbDims = 4;
        boxes_reshape_dims.d[0] = regression_box_dims.d[0];
        boxes_reshape_dims.d[1] = 4;
        boxes_reshape_dims.d[2] = reg_max;
        boxes_reshape_dims.d[3] = -1;
        Permutation boxes_reshape_permutation{0, 2, 1, 3};
        auto boxes_reshape_reshape = network->addShuffle(*regression_box->getOutput(0));
        boxes_reshape_reshape->setReshapeDimensions(boxes_reshape_dims);  ///< reshape to shape(1, 4, reg_max, anchors)
        boxes_reshape_reshape->setSecondTranspose(boxes_reshape_permutation);  ///< transpose to shape(1, reg_max, 4, anchors)
        auto dfl_softmax = network->addSoftMax(*boxes_reshape_reshape->getOutput(0)); ///< Perform softmax in 'reg_max' axes, shape(1, reg_max, 4, anchors)

        float* dfl_weights = new float[reg_max]{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0}; ///< [SAI-KEY]
        Weights emptywts{DataType::kFLOAT, nullptr, 0};
        Weights dfl_constant;
        dfl_constant.type = DataType::kFLOAT; ///< or kHALF
        dfl_constant.count = reg_max;
        dfl_constant.values = dfl_weights;
        auto dfl_box = network->addConvolutionNd(*dfl_softmax->getOutput(0), 1, DimsHW{1, 1}, dfl_constant, emptywts); ///< shape(1, 1, 4, anchors)

        Dims dfl_box_dims = dfl_box->getOutput(0)->getDimensions();
        Dims dfl_box_reshape_dims;
        dfl_box_reshape_dims.nbDims = 3;
        dfl_box_reshape_dims.d[0] = dfl_box_dims.d[0];
        dfl_box_reshape_dims.d[1] = 4;
        dfl_box_reshape_dims.d[2] = dfl_box_dims.d[3];
        auto box = network->addShuffle(*dfl_box->getOutput(0)); 
        box->setReshapeDimensions(dfl_box_reshape_dims); ///< shape(1, 4, anchors)

    // }

    // Dims box_dims = box->getOutput(0)->getDimensions();
    // printf("box_dims.ndims: %d \n", box_dims.nbDims);
    // for(int i=0; i<box_dims.nbDims; i++){
    //     printf("dim[%d]: %d \n", i, box_dims.d[i]);
    // }

    /** Resolve dbox results 
     * 1. dfl_conv: lt(shape(1, 2, M)), rb(shape(1, 2, M)).
     * 2. x1y1 = anchor_points - lt
     * 3. x2y2 = anchor_points + rb
     * 4. Merge as x1y1x2y2 format.
     * 5. x1y1x2y2 * stride
     * 6. shape(4, M)
     */
    /** Construct anchors grid for 3 feature maps */
    Weights stride_weights;
    stride_weights.type   = DataType::kFLOAT; ///< or kHALF
    stride_weights.count  = (FM0_H*FM0_W+FM1_H*FM1_W+FM2_H*FM2_W);
    stride_weights.values = stride;
    auto stride_constant = network->addConstant(Dims{3, {1, 1, (FM0_H*FM0_W+FM1_H*FM1_W+FM2_H*FM2_W)}}, stride_weights);

    Weights fm0_anchor_grid_weights;
    fm0_anchor_grid_weights.type   = DataType::kFLOAT; ///< or kHALF
    fm0_anchor_grid_weights.count  = FM0_H*FM0_W*2;
    fm0_anchor_grid_weights.values = fm0_anchor_grid;
    auto fm0_anchor_grid_constant = network->addConstant(Dims{3, {1, FM0_H*FM0_W, 2}}, fm0_anchor_grid_weights);

    Weights fm1_anchor_grid_weights;
    fm1_anchor_grid_weights.type   = DataType::kFLOAT; ///< or kHALF
    fm1_anchor_grid_weights.count  = FM1_H*FM1_W*2;
    fm1_anchor_grid_weights.values = fm1_anchor_grid;
    auto fm1_anchor_grid_constant = network->addConstant(Dims{3, {1, FM1_H*FM1_W, 2}}, fm1_anchor_grid_weights);

    Weights fm2_anchor_grid_weights;
    fm2_anchor_grid_weights.type   = DataType::kFLOAT; ///< or kHALF
    fm2_anchor_grid_weights.count  = FM2_H*FM2_W*2;
    fm2_anchor_grid_weights.values = fm2_anchor_grid;
    auto fm2_anchor_grid_constant = network->addConstant(Dims{3, {1, FM2_H*FM2_W, 2}}, fm2_anchor_grid_weights);

    ITensor* AnchorGridTensor[] = {fm0_anchor_grid_constant->getOutput(0), fm1_anchor_grid_constant->getOutput(0), fm2_anchor_grid_constant->getOutput(0)};
    auto AnchorGrid = network->addConcatenation(AnchorGridTensor, 3); ///< shape(1, anchors, 2)
    AnchorGrid->setAxis(1);
    auto AnchorGrid_2 = network->addShuffle(*AnchorGrid->getOutput(0));
    AnchorGrid_2->setFirstTranspose(Permutation{0, 2, 1}); ///< shape(1, 2, anchors)

    Dims dfl_conv_dim = box->getOutput(0)->getDimensions();
    auto box_lt = network->addSlice(*box->getOutput(0), Dims{3, {0, 0, 0}}, Dims{3, {dfl_conv_dim.d[0], 2, dfl_conv_dim.d[2]}}, Dims{3, {1, 1, 1}});
    auto box_rb = network->addSlice(*box->getOutput(0), Dims{3, {0, 2, 0}}, Dims{3, {dfl_conv_dim.d[0], 2, dfl_conv_dim.d[2]}}, Dims{3, {1, 1, 1}});

    // Dims AnchorGrid_2_dims = AnchorGrid_2->getOutput(0)->getDimensions();
    // printf("AnchorGrid_2_dims.ndims: %d \n", AnchorGrid_2_dims.nbDims);
    // for(int i=0; i<AnchorGrid_2_dims.nbDims; i++){
    //     printf("dim[%d]: %d \n", i, AnchorGrid_2_dims.d[i]);
    // }

    // Dims box_lt_dims = box_lt->getOutput(0)->getDimensions();
    // printf("box_lt_dims.ndims: %d \n", box_lt_dims.nbDims);
    // for(int i=0; i<box_lt_dims.nbDims; i++){
    //     printf("dim[%d]: %d \n", i, box_lt_dims.d[i]);
    // }

    auto box_lt_primitive = network->addElementWise(*AnchorGrid_2->getOutput(0), *box_lt->getOutput(0), ElementWiseOperation::kSUB); ///< anchor_grid - box_lt
    // box_lt_primitive->setName("box_lt_primitive");
    auto box_rb_primitive = network->addElementWise(*AnchorGrid_2->getOutput(0), *box_rb->getOutput(0), ElementWiseOperation::kSUM); ///< anchor_grid + box_rb
    // box_rb_primitive->setName("box_rb_primitive");


    ITensor* DboxTensor[] = {box_lt_primitive->getOutput(0), box_rb_primitive->getOutput(0)};
    auto dbox = network->addConcatenation(DboxTensor, 2); ///< shape(1, 4, M)
    dbox->setAxis(1);

    /**
     * OUTPUT
     */
    auto box_raw = network->addElementWise(*dbox->getOutput(0), *stride_constant->getOutput(0), ElementWiseOperation::kPROD); ///< * stride, shape(1, 4, anchors)
    auto cls_raw = network->addActivation(*cls->getOutput(0), ActivationType::kSIGMOID); ///< shape(1, 48, anchors)

    // auto box = network->addShuffle(*dfl_box->getOutput(0)); 
    // box->setReshapeDimensions(dfl_box_reshape_dims); ///< shape(1, 4, anchors)


    /**  
     * [SAI-KEY] Postprocess
    */
    /**
     * 先计算每一个cell对应的类别预测得分最大值，并获取相应的索引，该索引即当前cell所预测的类别
     * cls_top1_s1->getOutput(0): 每一个cell对应的最大得分值, kFLOAT
     * cls_top1_s1->getOutput(1): 每一个cell对应的最大得分值所对应的索引, 即当前cell属于哪一个类, kINT32, class indicator.
     */
    auto cls_top1_s1 = network->addTopK(*cls_raw->getOutput(0), TopKOperation::kMAX, 1, 0x02); ///< shape(1, 48, M)->shape(1, 1, anchors)
    // Dims cls_top1_s1_value_dim = cls_top1_s1->getOutput(0)->getDimensions();
    // Dims cls_top1_s1_index_dim = cls_top1_s1->getOutput(1)->getDimensions();

    // printf("cls_top1_s1->getOutput(0)->type: %d \n", cls_top1_s1->getOutput(0)->getType());
    // printf("cls_top1_s1_value_dim.ndims: %d \n", cls_top1_s1_value_dim.nbDims);
    // for(int i=0; i<cls_top1_s1_value_dim.nbDims; i++){
    //     printf("dim[%d]: %d \n", i, cls_top1_s1_value_dim.d[i]);
    // }

    // printf("cls_top1_s1->getOutput(1)->type: %d \n", cls_top1_s1->getOutput(1)->getType());
    // printf("cls_top1_s1_index_dim.ndims: %d \n", cls_top1_s1_index_dim.nbDims);
    // for(int i=0; i<cls_top1_s1_index_dim.nbDims; i++){
    //     printf("dim[%d]: %d \n", i, cls_top1_s1_index_dim.d[i]);
    // }

    /**
     * 执行topK操作，挑选出得分最大的前MAX_OUTPUT_BBOX_COUNT个cell
     * cls_top1_s2->getOutput(0): 得分值.
     * cls_top1_s2->getOutput(1): cell的索引.
     */
    auto cls_top1_s2 = network->addTopK(*cls_top1_s1->getOutput(0), TopKOperation::kMAX, NAMESPACE_YOLOv10::MAX_OUTPUT_BBOX_COUNT, 0x04); ///< shape(1, 1, M)->shape(1, 1, MAX_OUTPUT_BBOX_COUNT)
    // Dims cls_top1_s2_value_dim = cls_top1_s2->getOutput(0)->getDimensions(); ///< 得分
    Dims cls_top1_s2_index_dim = cls_top1_s2->getOutput(1)->getDimensions(); ///< 索引


    /** Gather class tensor*/
    Dims cls_top1_s2_return_1_reshape_dims;
    cls_top1_s2_return_1_reshape_dims.nbDims = 1;
    cls_top1_s2_return_1_reshape_dims.d[0] = cls_top1_s2_index_dim.d[2];
    auto class_gather_index = network->addShuffle(*cls_top1_s2->getOutput(1));
    class_gather_index->setReshapeDimensions(cls_top1_s2_return_1_reshape_dims);
    auto class_output = network->addGather(*cls_top1_s1->getOutput(1), *class_gather_index->getOutput(0), 2); ///< class tensor [SAI-KEY] Depends on TRT version, this is 8.0.1.
    class_output->setNbElementWiseDims(0);
    /** class_output为shape(1, 1, 100)的Tensor, 基于内存存储的规则，此处不需要执行Transpose操作. */

    /** Gather box tensor*/
    auto box_output_1 = network->addGather(*box_raw->getOutput(0), *class_gather_index->getOutput(0), 2); ///< class tensor [SAI-KEY] Depends on TRT version, this is 8.0.1.
    box_output_1->setNbElementWiseDims(0);

    auto box_output = network->addShuffle(*box_output_1->getOutput(0)); ///< Transpose to shape(MAX_OUTPUT_BBOX_COUNT, 4)
    box_output->setFirstTranspose(Permutation{0, 2, 1});
    /** box_output为shape(1, 100, 4)的Tensor. */

    Dims box_output_dim = box_output->getOutput(0)->getDimensions();
    printf("box_output->type: %d \n", box_output->getType());
    printf("box_output_dim.ndims: %d \n", box_output_dim.nbDims);
    for(int i=0; i<box_output_dim.nbDims; i++){
        printf("dim[%d]: %d \n", i, box_output_dim.d[i]);
    }

    /**
     * cls_top1_s2 : scores,  flaot, shape(1, MAX_OUTPUT_BBOX_COUNT)
     * class_output: classes, int32, shape(1, MAX_OUTPUT_BBOX_COUNT)
     * box_output_2: boxes,   float, shape(MAX_OUTPUT_BBOX_COUNT, 4)
     */
    /** OUTPUT: scores, float, shape(1, MAX_OUTPUT_BBOX_COUNT) */
    cls_top1_s2->getOutput(0)->setName(mScoresBlobName);
    network->markOutput(*cls_top1_s2->getOutput(0));

    /** OUTPUT: classes, int32, shape(1, MAX_OUTPUT_BBOX_COUNT) */
    class_output->getOutput(0)->setName(mClassesBlobName);
    network->markOutput(*class_output->getOutput(0));

    /** OUTPUT: box, float, shape(MAX_OUTPUT_BBOX_COUNT, 4) */
    box_output->getOutput(0)->setName(mBoxesBlobName);
    network->markOutput(*box_output->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB


    if(compute_mode == INT8){
        std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, yolov8_model_input_width, yolov8_model_input_height, "./coco_calib/", "int8calib.table", mInputBlobName);
        config->setInt8Calibrator(calibrator);
    }else if(compute_mode == FP16){
        config->setFlag(BuilderFlag::kFP16);
    }  

    SLOG_FMT_INFO("[EngineGen] Building engine, please wait for a while...");
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    SLOG_FMT_INFO("[EngineGen] Build engine successfully!");

    /** Don't need the network any more */
    network->destroy();

    /** Release host memory */
    for (auto &mem : weightMap){
        free((void *)(mem.second.values));
    }

    return engine;
}


void YOLOv10_Model::GenEngine(unsigned int maxBatchSize, IHostMemory** modelStream, std::string wts_file, float &gd, float &gw, ComputationMode compute_mode) 
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    ICudaEngine* engine = NULL;
    // Create model to populate the network, then set the outputs and create an engine
    engine = BuildEngine(maxBatchSize, builder, config, DataType::kFLOAT, wts_file, gd, gw, compute_mode);
    
    assert(engine != NULL);
    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    config->destroy();
    builder->destroy();
}


void YOLOv10_Model::DoInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, void** scores, void** classes, void** boxes, Dims& input_dims) 
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, mModelInputDataSize * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    context.setBindingDimensions(inputIndex, input_dims); ///< [SAI-KEY] Set input dimensions.
    CUDA_CHECK(cudaMemcpyAsync(scores[0],  buffers[scores_output_index],  mScoresOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(classes[0], buffers[classes_output_index], mClassesOutputSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(boxes[0],   buffers[boxes_output_index],   mBoxesOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}


int YOLOv10_Model::CreateEngine(std::string wts_file, ComputationMode compute_mode)
{
    IHostMemory* modelStream{nullptr};
    GenEngine(1, &modelStream, wts_file, global_depth, global_width, compute_mode);
    assert(modelStream != nullptr);
    std::ofstream p(EnginePath, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file: " << EnginePath << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0;
}


int YOLOv10_Model::Infer(std::string eval_src_dir, std::string eval_dst_dir)
{
    std::ifstream file(EnginePath, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /** 
     * Alloc CPU Memory 
     */
    /** Input image */
    static float* data;
    data = (float*)malloc(batchsize * mModelInputDataSize * sizeof(float));
    /** Scores  outputs. */
    static float* ScoresOutputFloat[1];
    ScoresOutputFloat[0] = (float*)malloc(batchsize * mScoresOutputSize * sizeof(float));
    /** Classes outputs. */
    static int32_t* ClassesOutputInt32[1];
    ClassesOutputInt32[0] = (int32_t*)malloc(batchsize * mClassesOutputSize * sizeof(int32_t));
    /** Boxes  outputs. */
    static float* BoxesOutputFloat[1];
    BoxesOutputFloat[0] = (float*)malloc(batchsize * mBoxesOutputSize * sizeof(float));
    

    /** 
     * Alloc GPU Memory 
     */
    /** Set buffer */
    void* buffers[4]; ///< Both input and output buffers.
    /** Get gpu buffer index */
    inputIndex           = engine->getBindingIndex(mInputBlobName);
    scores_output_index  = engine->getBindingIndex(mScoresBlobName);
    classes_output_index = engine->getBindingIndex(mClassesBlobName);
    boxes_output_index   = engine->getBindingIndex(mBoxesBlobName);
    /** Malloc gpu memory */
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex],           batchsize * mModelInputDataSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[scores_output_index],  batchsize * mScoresOutputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[classes_output_index], batchsize * mClassesOutputSize * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&buffers[boxes_output_index],   batchsize * mBoxesOutputSize * sizeof(float)));

#if 0
    /** [SAI-GDB] */
    Dims output_dims = engine->getBindingDimensions(boxes_output_index);
    SLOG_FMT_INFO("[EngineGen] output_dims.nbDims: %d", output_dims.nbDims);
    for(int i=0; i<output_dims.nbDims; i++){
        SLOG_FMT_INFO("[EngineGen] dim: %d", output_dims.d[i]);
    }
#endif

    printf("[SAI-INFER] Read txt image...\n");
    std::string ins;
    std::ifstream image_f("/zqpe/1009_TRT_YOLOv10/004_Evals/001_Src/txt/image.txt");
    int i = 0;
    for(int line=0; line<(3*384*640); line++){ ///< [SAI-KP]注意数据大小
        getline(image_f, ins);
        data[line] = (float)(atof(ins.c_str()));
        i++;
    }
    image_f.close();
    printf("[SAI-INFER] Do inference...\n");
    DoInference(*context, stream, buffers, data, (void**)ScoresOutputFloat, (void**)ClassesOutputInt32, (void**)BoxesOutputFloat, mInputDims);

    for(int s=0; s<100; s++){
        printf("%d: %f\n", s, BoxesOutputFloat[0][s]);
    }

    /** Release stream and buffers. */
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[scores_output_index]));
    CUDA_CHECK(cudaFree(buffers[classes_output_index]));
    CUDA_CHECK(cudaFree(buffers[boxes_output_index]));

    /** Destroy the engine. */
    context->destroy();
    engine->destroy();
    runtime->destroy();

    free(data);
    free(ScoresOutputFloat[0]);
    free(ClassesOutputInt32[0]);
    free(BoxesOutputFloat[0]);
}

/**
 * @brief 解析参数
 * 
 * @return int 0: 解析参数成功;
 *            -1: 参数文件不存在
 */
int YOLOv10_Model::ParseParams()
{
    IniParser info_parser;
    IniParser ini_parser;

    char BinPathBuffer[256] = {'\0'};
    readlink("/proc/self/exe", BinPathBuffer, sizeof(BinPathBuffer));
    std::string TrtInfoPath = "dais_definition.ini";

    ///
    char *dir_name = dirname(BinPathBuffer);
    std::string s_dir_name = std::string(dir_name);

    std::string definition_path = s_dir_name + "/dais_definition";

    printf("definition_path: %s\n", definition_path.c_str());

    /** dais_definition.ini */
    std::string TRT_INFO_PATH = ini_parser.GetIniFilePath(definition_path, TrtInfoPath);
    int ret = info_parser.ImportIni(TRT_INFO_PATH); ///< 解析配置文件
    if(ret == 0){
        SLOG_FMT_INFO("[PARSE_PARAM] Model info read failed, cannot find params file: %s", TRT_INFO_PATH.c_str());
    }  

    /** Parse *_params.ini file path */
    std::string trt_info_path;
    trt_info_path = info_parser.Parse("OPTIONS", "MODEL_PARAMS_FILE");
    if(trt_info_path == ""){
        std::cout << "Invalid model info file." << std::endl;
        SLOG_FMT_ERROR("[PARSE_PARAM] Model params file: %s", trt_info_path.c_str());
        return -1;
    }
    SLOG_FMT_INFO("[PARSE_PARAM] Model params file: %s", trt_info_path.c_str());
    std::string CONFIG_PATH = ini_parser.GetIniFilePath(definition_path, trt_info_path);

    /** [SAI-TODO] Chech whether config file is exists. */
    ini_parser.ImportIni(CONFIG_PATH); ///< 解析配置文件
    if(ret == 0)
    {
        std::cout << "Model ini file read failed" << std::endl;
    }  

    std::string str_value;
    str_value = ini_parser.Parse("COMMON_CONFIG", "ENGINE_PATH");
    if(str_value != ""){
        EnginePath = str_value;
    }

    std::cout << "Parse YOLOv10 params" << std::endl;
    str_value = ini_parser.Parse("YOLOv10", "SUB_EDITION");
    if(str_value != ""){
        yolov8_sub_edition = str_value;
    }

    if(yolov8_sub_edition == "l"){
        global_depth = 1.0;
        global_width = 1.0;
    }if(yolov8_sub_edition == "m"){
        global_depth = 0.67;
        global_width = 0.75;
    }if(yolov8_sub_edition == "n"){
        global_depth = 0.33;
        global_width = 0.25;
    }

    str_value = ini_parser.Parse("YOLOv10", "NUM_CLASSES");
    if(str_value != ""){
        yolov8_num_classes = atoi(str_value.c_str());
    }

    str_value = ini_parser.Parse("YOLOv10", "INPUT_WIDTH");
    if(str_value != ""){
        yolov8_model_input_width = atoi(str_value.c_str());
    }

    str_value = ini_parser.Parse("YOLOv10", "INPUT_HEIGHT");
    if(str_value != ""){
        yolov8_model_input_height = atoi(str_value.c_str());
    }

    str_value = ini_parser.Parse("YOLOv10", "BBOX_CONF");
    if(str_value != ""){
        yolov8_bbox_conf_thresh = atof(str_value.c_str());
    }
    str_value = ini_parser.Parse("YOLOv10", "NMS_THRESH");
    if(str_value != ""){
        yolov8_nms_thresh = atof(str_value.c_str());
    }
    

#if 1
    SLOG_FMT_INFO("[EngineGen] ENGINE_PATH: %s",    EnginePath.c_str());
    SLOG_FMT_INFO("[EngineGen] SUB_EDITION: %s",    yolov8_sub_edition.c_str());
    SLOG_FMT_INFO("[EngineGen] DEPTH_MULTIPLE: %f", global_depth);
    SLOG_FMT_INFO("[EngineGen] WIDTH_MULTIPLE: %f", global_width);
    SLOG_FMT_INFO("[EngineGen] NUM_CLASSES: %d",    yolov8_num_classes);
    SLOG_FMT_INFO("[EngineGen] INPUT_WIDTH: %d",    yolov8_model_input_width);
    SLOG_FMT_INFO("[EngineGen] INPUT_HEIGHT: %d",   yolov8_model_input_height);
    SLOG_FMT_INFO("[EngineGen] BBOX_CONF: %f",      yolov8_bbox_conf_thresh);
    SLOG_FMT_INFO("[EngineGen] NMS_THRESH: %f",     yolov8_nms_thresh);
#endif


    return 0;
}
