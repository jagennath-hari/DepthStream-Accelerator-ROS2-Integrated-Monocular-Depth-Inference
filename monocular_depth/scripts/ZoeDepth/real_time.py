import cv2
import torch
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
import time

class webcamDepth():
    def __init__(self):
        # Initialize the webcam feed.
        self.cap = cv2.VideoCapture(0)
        self.model_zoe_nk = build_model(get_config("zoedepth", "infer")).to("cuda:0")

    def stream(self):
        torch.cuda.synchronize()
        with torch.no_grad():  # Disable gradient calculations for faster inference
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Convert the image from BGR to RGB
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert the numpy image to a torch tensor
                input_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()

                # Optionally, normalize the image (if the model requires normalization)
                input_tensor = input_tensor / 255.0  # If needed, make sure the model expects this normalization

                # Add a batch dimension (BCHW) and transfer the image to the GPU
                input_tensor = input_tensor.unsqueeze(0).to("cuda:0")

                # Perform inference
                start = time.time()
                depth_tensor = self.model_zoe_nk.infer(input_tensor)
                #print(self.model_zoe_nk(input_tensor))
                print((time.time() - start ) * 1000)

                # Convert the tensor to a numpy array, if necessary for visualization
                depth_numpy = depth_tensor.squeeze().cpu().numpy()  # Modify if the dimensionality doesn't match
                print("Depth: ")
                print(depth_numpy.shape)
                colored_depth = colorize(depth_numpy)

                # Show the colorized depth map
                cv2.imshow("Depth Map", colored_depth)

                # Exit loop on 'q' keypress
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    depth_stream = webcamDepth()
    depth_stream.stream()

if __name__ == "__main__":
    main()
