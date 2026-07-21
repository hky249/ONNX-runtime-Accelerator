import onnxruntime
import numpy as np
import json
import os
import urllib.request

print("="*50)
print("test real image: test_image.jpg with ONNX model")
print("="*50)

print("="*50)
print("loading settings")
print("="*50)

with open("onnx_models/preprocess_info.json", "r") as f:
    preprocess_info = json.load(f)

print(f"model name: {preprocess_info['model_name']}")
print(f"mean: {preprocess_info['mean']}")
print(f"std: {preprocess_info['std']}")
print(f"input_size: {preprocess_info['input_size']}")

print("="*50)
print("loading ONNX model")
print("="*50)

ort_session = onnxruntime.InferenceSession("onnx_models/resnet18.onnx", providers=['CPUExecutionProvider'])

print("ONNX model loaded: ")
print(f"- input nodes: {ort_session.get_inputs()[0].name}")
print(f"- output nodes: {ort_session.get_outputs()[0].name}")