#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/cpu_provider_factory.h>

const std::string MODEL_PATH = "../onnx_models/resnet18.onnx";
const std::string LABELS_PATH = "../imagenet_classes.txt";
const std::string IMAGE_PATH = "../test_image.jpg";

const int INPUT_WIDTH = 224;
const int INPUT_HEIGHT = 224;
const int NUM_CLASSES = 1000;

const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> STD = {0.229f, 0.224f, 0.225f};

int main() {
	return 0;
}