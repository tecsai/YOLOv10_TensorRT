#ifndef __YOLOV5_MODEL_H__
#define __YOLOV5_MODEL_H__


#include "cuda_runtime_api.h"
#include "logging.h"
#include "calibrator.h"
#include "Model.h"
#include "YOLOv10_Common.h"
#include "ModelConfig.h"
#include "FF_DirFile.h"
#include "FF_IniParser.h"
#include "SLogger.h"
#include "FF_Path.h"

using namespace nvinfer1;


class YOLOv10_Model: public Model
{
    /** Feature map resolution calculated based on 640*386. */
    #define FM0_H 48
    #define FM0_W 80
    #define FM1_H 24
    #define FM1_W 40
    #define FM2_H 12
    #define FM2_W 20

public:
    YOLOv10_Model();
    ~YOLOv10_Model();

    virtual int CreateEngine(std::string wts_file, ComputationMode compute_mode=FP16);
    virtual int Infer(std::string eval_src_dir, std::string eval_dst_dir);

private:
    /**
     * @brief Construct network based TensorRT.
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
    ICudaEngine* BuildEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, 
                                        std::string wts_file, float &gd, float &gw, ComputationMode compute_mode);
    
    /** Generate inference engine.
     * @brief 
     * @param maxBatchSize 
     * @param modelStream 
     * @param model_dir 
     * @param compute_mode 
     */
    void GenEngine(unsigned int maxBatchSize, IHostMemory** modelStream, std::string wts_file, float &gd, float &gw, ComputationMode compute_mode);
    
    /**
     * @brief Parsing imported parameters from ini file.
     * 
     * @return int 0: Success;
     *            -1: Fail;
     */
    int ParseParams();
    
    /**
     * @brief Do inference.
     * @param context 
     * @param stream 
     * @param buffers 
     * @param input 
     * @param output 
     * @param batchSize 
     */
    void DoInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, void** scores, void** classes, void** boxes, Dims& input_dims);


    /** Get width and depth for network construction. */
    int get_width(int x, float gw, int divisor = 8);
    int get_depth(int x, float gd);


private:
    int batchsize;
    int inputIndex;
    int scores_output_index;
    int classes_output_index;
    int boxes_output_index;

    Dims mInputDims;

    /** Input and output buffer size. */
    int mModelInputDataSize;
    int mScoresOutputSize;
    int mClassesOutputSize;
    int mBoxesOutputSize;

    /** Input and output blob name. */
    char* mInputBlobName;
    char* mBoxesBlobName;
    char* mScoresBlobName;
    char* mClassesBlobName;
    
    float global_depth = 0.0;
    float global_width = 0.0;

    /** TensorRT */
    Logger gLogger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::string ModelPath;
    std::string EnginePath;


    float fm0_anchor_grid[2*FM0_H*FM0_W];
    float fm1_anchor_grid[2*FM1_H*FM1_W];
    float fm2_anchor_grid[2*FM2_H*FM2_W];

    float stride[FM0_H*FM0_W+FM1_H*FM1_W+FM2_H*FM2_W];

    int reg_max;
};





#endif ///< __YOLOV5_MODEL_H__