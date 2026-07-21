import onnxruntime
import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms
import os

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

print("ONNX model loaded")

print("="*50)
print("preprocessing test image")
print("="*50)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=preprocess_info['mean'], std=preprocess_info['std'])
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor.numpy()

print("="*50)
print("loading ImageNet class labels")
print("="*50)

labels_path = "imagenet_classes.txt"
if not os.path.exists(labels_path):
    try:
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, labels_path)
        print(f"Downloaded ImageNet class labels to {labels_path}")
    except Exception as e:
        print(f"Error occurred while downloading ImageNet class labels: {e}")
        labels = None
else:
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"Loaded ImageNet class labels from {labels_path}")

print("="*50)
print("running inference on test image")
print("="*50)

test_image_path = "test_image.jpg"
if not os.path.exists(test_image_path):
    print(f"Test image {test_image_path} not found. Please provide a valid image.")
    exit(1)

input_data = preprocess_image(test_image_path)

ort_input = {ort_session.get_inputs()[0].name: input_data}
output = ort_session.run(None, ort_input)

print(f"ONNX Runtime output shape: {output[0].shape}")

print("="*50)
print("predictions")
print("="*50)

probs = np.exp(output[0]) / np.sum(np.exp(output[0]))
top5_indices = np.argsort(probs[0])[::-1][:5]

print("Top 5 predictions:")
for idx in top5_indices:
    label = labels[idx] if labels else f"Class {idx}"
    probability = probs[0][idx]
    print(f"{label}: {probability:.4f}")