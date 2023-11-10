import os
import sys
import argparse

import numpy as np
import tensorrt as trt
import time
import torch

import numpy as np
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
import cv2
#from zoedepth.utils.misc import colorize

class TensorRTInfer:

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            print(f"Binding {i}: Name={name}, dtype={dtype}, shape={shape}")



    def input_spec(self):
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        return self.outputs[1]['shape'], self.outputs[1]['dtype']

    def infer(self, batch, top=1):
        # Prepare the output data
        output = np.zeros(*self.output_spec())

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        cuda.memcpy_dtoh(output, self.outputs[1]['allocation'])
        return output


def main():
    trt_infer = TensorRTInfer("/home/hari/ZoeDepth/test.trt")
    input_shape, _ = trt_infer.input_spec()
    mat = cv2.imread("/home/hari/my_photo-1.jpg")
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    mat = cv2.resize(mat, dsize = (512, 384))
    mat = np.transpose(mat, (2, 0, 1))
    mat = np.expand_dims(mat, axis=0)
    normalized_image = mat.astype(np.float32) / 255.0
    batch_images = normalized_image


    # Run inference
    output = trt_infer.infer(batch_images)

    # Print the results
    print("Inference Output:")
    print(output)

if __name__ == "__main__":
    main()
