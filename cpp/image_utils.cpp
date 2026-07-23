#include "image_utils.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>

std::vector<float> ImageUtils::preprocess(
    const std::string& image_path,
    const std::vector<float>& mean,
    const std::vector<float>& std,
    int input_width,
    int input_height
) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to read image from path: " + image_path);
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    int width = image.cols;
    int height = image.rows;
    int new_width, new_height;
    if (width > height) {
        new_width = 256;
        new_height = static_cast<int>(height * (256.0 / width));
    } else {
        new_height = 256;
        new_width = static_cast<int>(width * (256.0 / height));
    }

    cv::resize(image, image, cv::Size(new_width, new_height));

    int crop_x = (new_width - input_width) / 2;
    int crop_y = (new_height - input_height) / 2;
    cv::Rect crop_rect(crop_x, crop_y, input_width, input_height);

    image = image(crop_rect);

    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    std::vector<float> input_data;
    input_data.reserve(input_width * input_height * 3);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_height; ++h) {
            for (int w = 0; w < input_width; ++w) {
                float pixel_value = image.at<cv::Vec3f>(h, w)[c];
                pixel_value = (pixel_value - mean[c]) / std[c];
                input_data.push_back(pixel_value);
            }
        }
    }

    return input_data;
}

std::vector<std::pair<int, float>> ImageUtils::postprocess(
    const std::vector<float>& output,
    int top_k
) {
    std::vector<float> softmax_output(output.size());
    float max_value = *std::max_element(output.begin(), output.end());
    float sum = 0.0f;

    for (size_t i = 0; i < output.size(); ++i) {
        softmax_output[i] = std::exp(output[i] - max_value);
        sum += softmax_output[i];
    }

    for (size_t i = 0; i < softmax_output.size(); ++i) {
        softmax_output[i] /= sum;
    }

    std::vector<int> indices(softmax_output.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                      [&softmax_output](int a, int b) {
                          return softmax_output[a] > softmax_output[b];
                      });
    
    std::vector<std::pair<int, float>> top_k_results;
    for (int i = 0; i < top_k; ++i) {
        int idx = indices[i];
        top_k_results.emplace_back(idx, softmax_output[idx]);
    }
    return top_k_results;
}

std::vector<std::string> ImageUtils::loadLabels(const std::string& labels_path) {
    std::vector<std::string> labels;
    std::ifstream file(labels_path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open labels file: " + labels_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    return labels;
}