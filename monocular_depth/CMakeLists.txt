cmake_minimum_required(VERSION 3.8)
project(monocular_depth)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find TensorRT headers and libraries.
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 /usr/lib/x86_64-linux-gnu)
message(STATUS "Found TensorRT libs ${TENSORRT_LIBRARY_INFER}")

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS $ENV{TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 /usr/lib/x86_64-linux-gnu)
message(STATUS "Found TensorRT plugin libs ${TENSORRT_LIBRARY_INFER_PLUGIN}")

# Assume that all TensorRT libraries are found in the same directory.
get_filename_component(TENSORRT_LIBRARY_DIR ${TENSORRT_LIBRARY_INFER} DIRECTORY)
link_directories(${TENSORRT_LIBRARY_DIR})

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/monocular_depth)
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

add_executable(depth_map src/depthMap.cpp)
ament_target_dependencies(depth_map
  rclcpp
  cv_bridge
  sensor_msgs
  std_msgs
  OpenCV
)

# TensorRT libraries are linked separately
target_link_libraries(depth_map
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY_INFER}
  ${TENSORRT_LIBRARY_INFER_PLUGIN}
)

# Install C++ executable
install(TARGETS
  depth_map
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
