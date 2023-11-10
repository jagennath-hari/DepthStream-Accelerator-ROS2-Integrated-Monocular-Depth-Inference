#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import torch
import PIL
import numpy as np
import matplotlib

class monoDepth(Node):
    def __init__(self):
        super().__init__("mono_depth")
        self.bridge = CvBridge()
        self.zoe_ = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
        self.imageSub_ = self.create_subscription(Image, "/zed2i/zed_node/rgb/image_rect_color", self.cameraCallback_, 1)
        self.depthSub_ = self.create_subscription(Image, "/zed2i/zed_node/depth/depth_registered", self.depthCallback_, 1)
        self.f = 1066.1031494140625
        self.cx = 965.7115478515625
        self.cy = 547.1150512695312

    def colorize(self, value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
        """Converts a depth map to a color image.

        Args:
            value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
            vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
            vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
            cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
            invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
            invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
            background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
            gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
            value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

        Returns:
            numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        value = value.squeeze()
        if invalid_mask is None:
            invalid_mask = value == invalid_val
        mask = np.logical_not(invalid_mask)

        # normalize
        vmin = np.percentile(value[mask],2) if vmin is None else vmin
        vmax = np.percentile(value[mask],85) if vmax is None else vmax
        if vmin != vmax:
            value = (value - vmin) / (vmax - vmin)  # vmin..vmax
        else:
            # Avoid 0-division
            value = value * 0.

        # squeeze last dim if it exists
        # grey out the invalid values

        value[invalid_mask] = np.nan
        cmapper = matplotlib.cm.get_cmap(cmap)
        if value_transform:
            value = value_transform(value)
            # value = value / value.max()
        value = cmapper(value, bytes=True)  # (nxmx4)

        # img = value[:, :, :]
        img = value[...]
        img[invalid_mask] = background_color

        #     return img.transpose((2, 0, 1))
        if gamma_corrected:
            # gamma correction
            img = img / 255
            img = np.power(img, 2.2)
            img = img * 255
            img = img.astype(np.uint8)
        return img

    def cameraCallback_(self, msg):
        try:
            torch.cuda.synchronize()
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pillow_img = PIL.Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            depth_numpy = self.zoe_.infer_pil(pillow_img)
            valid_depth_map = depth_numpy[np.isfinite(depth_numpy)]
            average_depth = np.mean(valid_depth_map)
            self.get_logger().info("DEPTH NEURAL: {}".format(average_depth))
            depth_img = self.colorize(depth_numpy)
            resized = cv2.resize(depth_img, dsize=(0, 0), dst = None, fx = 0.5, fy = 0.5)
            cv2.imshow("FRAME", resized)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error("%s" % e)

    def depthCallback_(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            cv_image_array = np.array(cv_image, dtype=np.float32)
            valid_depth_map = cv_image_array[np.isfinite(cv_image_array)]
            average_depth = np.mean(valid_depth_map)
            self.get_logger().info("DEPTH ZED: {}".format(average_depth))
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