import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.models.zoedepth_nk import ZoeDepthNK
import torch.onnx
import tensorrt as trt

class convert():
    def __init__(self):
        self.model_zoe_nk = build_model(get_config("zoedepth_nk", "infer"))

    def start(self):
        #print(self.model_zoe_nk.modules)
        torch.onnx.export(self.model_zoe_nk, torch.rand(1, 3, 376,  672), "zoe_nk.onnx", input_names=['input'],
                  output_names=['output'], export_params=True)

def main():
    process_ = convert()
    process_.start()

if __name__ == "__main__":
    main()