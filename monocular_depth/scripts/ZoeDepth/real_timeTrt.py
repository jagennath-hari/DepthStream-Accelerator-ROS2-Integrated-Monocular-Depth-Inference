from trt2 import TensorRTInfer
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
from zoedepth.utils.misc import colorize
import time

class depthEstimation(TensorRTInfer):
    def __init__(self, engine_path):
        super().__init__(engine_path)
        self.cap = cv2.VideoCapture(0)

    def stream(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert the image from BGR to RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the numpy image to a torch tensor
            mat = cv2.resize(rgb_image, dsize = (512, 384))
            mat = np.transpose(mat, (2, 0, 1))
            mat = np.expand_dims(mat, axis=0)
            normalized_image = mat.astype(np.float32) / 255.0

            start = time.time()
            output = self.infer(normalized_image)
            print((time.time() - start ) * 1000)
            #print(self.model_zoe_nk(input_tensor))

            # Convert the tensor to a numpy array, if necessary for visualization
            squeezed_image = np.squeeze(output)
            colored_depth = colorize(squeezed_image)

            # Show the colorized depth map
            cv2.imshow("Depth Map", colored_depth)

            # Exit loop on 'q' keypress
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    depth_ = depthEstimation("/home/hari/ZoeDepth/test.trt")
    depth_.stream()

if __name__ == "__main__":
    main()