#!/usr/bin/python3

from typing import List
import rclpy
from rclpy.context import Context
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
import cv2
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from ZoeDepth.trt2 import TensorRTInfer
import numpy as np
from ZoeDepth.zoedepth.utils.misc import colorize
import time

class monoDepth(Node):
    def __init__(self):
        super().__init__("mono_depth")
        self.bridge = CvBridge()
        self.trtInfer_ = TensorRTInfer("/home/hari/ZoeDepth/zoedepth_engine.trt")
        self.imageSub_ = self.create_subscription(Image, "/zed2i/zed_node/rgb/image_rect_color", self.cameraCallback_, 1)

    def cameraCallback_(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            mat = cv_image
            mat = np.transpose(mat, (2, 0, 1))
            mat = np.expand_dims(mat, axis=0)
            normalized_image = mat.astype(np.float32) / 255.0
            output = self.trtInfer_.infer(normalized_image)
            dept_map = np.squeeze(output)
            self.get_logger().info("Avg Depth: {}".format(np.mean(dept_map)))
            colored_depth = colorize(dept_map)
            cv2.imshow("Depth Map", colored_depth)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error("%s" % e)

def main():
    rclpy.init()
    monoDepth_ = monoDepth()
    rclpy.spin(monoDepth_)
    monoDepth_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()