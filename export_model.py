import torch
import torchvision.models as models
import torchvision.transforms as transforms
import onnx
import onnxruntime
import numpy as np
from PIL import Image
import os
import json

print("="*50)
print("loading pre-trained model...")
print("="*50)

weight = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weight)
model.eval()

print("model loaded")
print("model structure: ResNet18")
print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

#dummy input
print("="*50)
print("creating dummy input")
print("="*50)

batch_size = 1
dummy_input = torch.randn(batch_size, 3, 224, 224)

print(f"dummy input shape: {dummy_input.shape}")
print(f"dummy input dtype: {dummy_input.dtype}")

print("="*50)
print("exporting model to ONNX format")
print("="*50)

os.makedirs("onnx_models", exist_ok=True)

onnx_model_path = "onnx_models/resnet18.onnx"

torch.onnx.export(model,
                  dummy_input,
                  onnx_model_path,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

print(f"model exported to {onnx_model_path}")
print(f"file size: {os.path.getsize(onnx_model_path) / (1024 * 1024):.2f} MB")

print("="*50)
print("verifying the exported ONNX model")
print("="*50)

try:
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")
except onnx.onnx_cpp2py_export.checker.ValidationError as e:
    print(f"ONNX model is invalid: {e}")

print("model details:")
print(f"- Opset version: {onnx_model.opset_import[0].version}")
print(f"- Number of input nodes: {len(onnx_model.graph.input)}")
print(f"- Number of output nodes: {len(onnx_model.graph.output)}")
print(f"- Number of nodes in the graph: {len(onnx_model.graph.node)}")

print(("input details: "))
for inp in onnx_model.graph.input:
    print(f"- Name: {inp.name}")
    print(f"- Type: {inp.type}")

print(("output details: "))
for out in onnx_model.graph.output:
    print(f"- Name: {out.name}")
    print(f"- Type: {out.type}")

print("="*50)
print("verify results between PyTorch and ONNX Runtime")
print("="*50)

with torch.no_grad():
    pytorch_output = model(dummy_input)
print(f"PyTorch output shape: {pytorch_output.shape}")

ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

ort_input = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

ort_output = ort_session.run(None, ort_input)
onnx_output = ort_output[0]

print(f"ONNX Runtime output shape: {onnx_output.shape}")

max_abs_diff = np.max(np.abs(pytorch_output.numpy() - onnx_output))
mean_abs_diff = np.mean(np.abs(pytorch_output.numpy() - onnx_output))

print(f"Max absolute difference: {max_abs_diff:.6f}")
print(f"Mean absolute difference: {mean_abs_diff:.6f}")

if max_abs_diff < 1e-5:
    print("The outputs from PyTorch and ONNX Runtime are similar.")
else:
    print("The outputs from PyTorch and ONNX Runtime differ significantly.")

print("="*50)
print("save preprocess info")
print("="*50)

info = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'input_size': [224, 224],
    'num_classes': 1000
}

with open("onnx_models/preprocess_info.json", "w") as f:
    json.dump(info, f, indent=2)

print("preprocess info saved to onnx_models/preprocess_info.json")
print(f"- mean: {info['mean']}")
print(f"- std: {info['std']}")
print(f"- input_size: {info['input_size']}")