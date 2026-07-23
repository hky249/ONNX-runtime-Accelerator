#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>

class ImageUtils {
public:
    /**
     * @param image_path
     * @param mean
     * @param std
     * @param input_width
     * @param input_height
     * @return
     */
    static std::vector<float> preprocess(
        const std::string& image_path,
        const std::vector<float>& mean,
        const std::vector<float>& std,
        int input_width = 224,
        int input_height = 224
    );
    
    /**
     * @param output
     * @param top_k
     * @return
     */
    static std::vector<std::pair<int, float>> postprocess(
        const std::vector<float>& output,
        int top_k = 5
    );
    
    /**
     * @param labels_path
     * @return
     */
    static std::vector<std::string> loadLabels(const std::string& labels_path);
};


#endif