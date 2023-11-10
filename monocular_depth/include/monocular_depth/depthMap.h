#ifndef DEPTHMAP_H_
#define DEPTHMAP_H_

// ROS Includes
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.h>

// OpenCV Include
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

// CUDA TensorRT Includes
#include <cuda_runtime_api.h>
#include <NvInfer.h>

// Includes
#include <iostream>
#include <fstream> 
#include <memory>

class Logger : public nvinfer1::ILogger
{
private:
    rclcpp::Logger rosLogger_;
public:
    Logger() : rosLogger_(rclcpp::get_logger("mono_depth")) {}
    void log(Severity severity, const char* msg) noexcept override {
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            RCLCPP_ERROR(this->rosLogger_, "NvInfer error: %s", msg);
        }
    }
} gLogger;

class depthMap : public rclcpp::Node
{
private:
    // Params
    bool gui_;

    // ROS Functions
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageDataSub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depthMapPub_;
    void imageCallback_(const sensor_msgs::msg::Image & msg);
    
    // TensorRT and CUDA members
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;
    
    // Host and device buffers
    std::vector<float> inputHostBuffer_;
    std::vector<float> outputHostBuffer_;
    std::vector<float> depthHostBuffer_;
    void* inputDeviceBuffer_;
    void* outputDeviceBuffer_;
    void* depthDeviceBuffer_;
    
    // Buffer sizes
    int inputSize_;
    int outputSize_;
    int depthSize_;

    // TensorRT functions
    bool loadTRTEngine_(const std::string& engineFilePath);
    cv::Mat performInference_(const cv::Mat& inputMat);
    
    // Helper Functions
    int calculateVolume_(const nvinfer1::Dims& d);
    void matToVector_(const cv::Mat& mat, std::vector<float>& vec);
    
    // Viz
    cv::Mat showMap_(const cv::Mat& depth);

    sensor_msgs::msg::Image genMsg_(const cv::Mat &depthMap, const sensor_msgs::msg::Image & msg);

public:
    // Constructor and Deconstructor
    depthMap();
    ~depthMap();
};

#endif