#include "depthMap.h"

depthMap::depthMap() : Node("mono_depth")
{
    // Launch params get
    this->declare_parameter("trt_path", "./zoe.trt");
    this->declare_parameter("image_topic", "/image_rect_color");
    this->declare_parameter("gui", false);
    this->gui_ = this->get_parameter("gui").as_bool();

    // Initialize CUDA stream
    cudaStreamCreate(&(this->stream_));

    if (!this->loadTRTEngine_(this->get_parameter("trt_path").as_string())) RCLCPP_ERROR(this->get_logger(), "Failed to load TensorRT engine.");

    // Initialize input and output buffers
    this->inputHostBuffer_.resize(this->inputSize_ / sizeof(float));
    this->outputHostBuffer_.resize(this->outputSize_ / sizeof(float));
    this->depthHostBuffer_.resize(this->depthSize_ / sizeof(float));

    // Allocate GPU memory for input and output
    cudaMalloc(&(this->inputDeviceBuffer_), this->inputSize_);
    cudaMalloc(&(this->outputDeviceBuffer_), this->outputSize_);
    cudaMalloc(&(this->depthDeviceBuffer_), this->depthSize_);

    // Create CUDA stream
    cudaStreamCreate(&(this->stream_));

    // ROS Subscriber
    this->imageDataSub_ = this->create_subscription<sensor_msgs::msg::Image>(this->get_parameter("image_topic").as_string(), rclcpp::QoS(10), std::bind(&depthMap::imageCallback_, this, std::placeholders::_1));
    this->depthMapPub_ = this->create_publisher<sensor_msgs::msg::Image>("/mono_depth/depthMap", rclcpp::QoS(10));
}

depthMap::~depthMap()
{
    // Free GPU memory
    cudaFree(inputDeviceBuffer_);
    cudaFree(outputDeviceBuffer_);
    cudaFree(depthDeviceBuffer_);

    // Destroy TensorRT objects
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();

    // Destroy OpenCV Window
    cv::destroyAllWindows();
}

int depthMap::calculateVolume_(const nvinfer1::Dims& d)
{
    int vol = 1;
    for (int i = 0; i < d.nbDims; ++i) vol *= d.d[i];
    return vol;
}

bool depthMap::loadTRTEngine_(const std::string& engineFilePath)
{
    std::ifstream engineFile(engineFilePath, std::ios::binary);
    if (!engineFile)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to open engine file: %s", engineFilePath.c_str());
        return false;
    }
    engineFile.seekg(0, std::ifstream::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);
    std::vector<char> engineData(fileSize);
    engineFile.read(engineData.data(), fileSize);
    engineFile.close();

    // Deserialize the engine
    this->runtime_ = nvinfer1::createInferRuntime(gLogger);
    this->engine_ = runtime_->deserializeCudaEngine(engineData.data(), fileSize, nullptr);
    if (!this->engine_)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to create CUDA engine");
        return false;
    }

    this->context_ = engine_->createExecutionContext();
    if (!this->context_)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to create execution context");
        return false;
    }

    // Allocate buffers based on the size calculated from the engine bindings
    this->inputSize_ = this->calculateVolume_(engine_->getBindingDimensions(0)) * sizeof(float);
    this->outputSize_ = this->calculateVolume_(engine_->getBindingDimensions(1)) * sizeof(float);
    this->depthSize_ = this->calculateVolume_(engine_->getBindingDimensions(2)) * sizeof(float);

    // Resize host vectors
    this->inputHostBuffer_.resize(this->inputSize_ / sizeof(float));
    this->outputHostBuffer_.resize(this->outputSize_ / sizeof(float));
    this->depthHostBuffer_.resize(this->depthSize_ / sizeof(float));

    // Allocate device memory
    cudaMalloc(&this->inputDeviceBuffer_, this->inputSize_);
    cudaMalloc(&this->outputDeviceBuffer_, this->outputSize_);
    cudaMalloc(&this->depthDeviceBuffer_, this->depthSize_);

    RCLCPP_INFO(this->get_logger(), "Loaded Engine file %s", engineFilePath.c_str());
    return true;
}

void depthMap::matToVector_(const cv::Mat& mat, std::vector<float>& vec)
{
    if (mat.isContinuous()) vec.assign((float*)mat.datastart, (float*)mat.dataend);
    else for (int i = 0; i < mat.rows; ++i) vec.insert(vec.end(), mat.ptr<float>(i), mat.ptr<float>(i) + mat.cols);
}

cv::Mat depthMap::performInference_(const cv::Mat& inputMat)
{
    // Assuming each pixel in the input is a float and the GpuMat is continuous
    if (!inputMat.isContinuous())
    {
        RCLCPP_ERROR(this->get_logger(), "Input GpuMat is not continuous.");
        return inputMat;
    }

    // Creating the input
    std::vector<float> inputImage;
    this->matToVector_(inputMat, inputImage);

    // Copy input from GpuMat to the input device buffer
    cudaMemcpyAsync(
            this->inputDeviceBuffer_, 
            inputImage.data(), 
            this->inputSize_, 
            cudaMemcpyHostToDevice, 
            this->stream_
    );

    // Execute the network
    void* buffers[] = {this->inputDeviceBuffer_, this->outputDeviceBuffer_, this->depthDeviceBuffer_};
    this->context_->enqueueV2(buffers, this->stream_, nullptr);

    // Copy the output from the output device buffer to the output host buffer
    cudaMemcpyAsync(
        this->outputHostBuffer_.data(), 
        this->outputDeviceBuffer_, 
        this->outputSize_, 
        cudaMemcpyDeviceToHost, 
        this->stream_
    );

    // Copy the depth information from the depth device buffer to the depth host buffer
    cudaMemcpyAsync(
        this->depthHostBuffer_.data(), 
        this->depthDeviceBuffer_, 
        this->depthSize_, 
        cudaMemcpyDeviceToHost, 
        this->stream_
    );

    // Wait for all CUDA operations to finish
    cudaStreamSynchronize(this->stream_);

    // Construct Depth Map
    cv::Mat finalOutput(384, 672, CV_32F, this->depthHostBuffer_.data());

    return finalOutput;
}

cv::Mat depthMap::showMap_(const cv::Mat& depth)
{
    // Normalize the depth map to 0-1
    cv::Mat normalizedDepthMap;
    cv::normalize(depth, normalizedDepthMap, 0.0f, 1.0f, cv::NORM_MINMAX);

    // Invert the depth values
    cv::Mat invertedDepthValues = 1.0f - normalizedDepthMap;

    // Convert to 8-bit image
    cv::Mat invertedDepthMap8U;
    invertedDepthValues.convertTo(invertedDepthMap8U, CV_8U, 255.0);

    // Apply the inferno colormap
    cv::Mat infernoDepthMap;
    cv::applyColorMap(invertedDepthMap8U, infernoDepthMap, cv::COLORMAP_MAGMA);

    return infernoDepthMap;
}

sensor_msgs::msg::Image depthMap::genMsg_(const cv::Mat &depthMap, const sensor_msgs::msg::Image & msg)
{
    cv_bridge::CvImage cvImage(msg.header, sensor_msgs::image_encodings::TYPE_32FC1, depthMap);
    
    sensor_msgs::msg::Image rosImage;
    cvImage.toImageMsg(rosImage);
    
    return rosImage;
}

void depthMap::imageCallback_(const sensor_msgs::msg::Image & msg)
{
    try
    {
        cv::Mat frame = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(frame.cols, frame.rows), cv::Scalar(), true, false);

        cv::Mat depth = this->performInference_(blob);
        this->depthMapPub_->publish(this->genMsg_(depth, msg));
        if (this->gui_)
        {
            cv::Mat coloredDepth = this->showMap_(depth);
            cv::imshow("Depth Map", coloredDepth);
            cv::waitKey(1);
        }
    }
    catch(cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "Cv Bridge error: %s", e.what());
    }
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<depthMap>());
    rclcpp::shutdown();

    return 0;
}