#include <opencv2/opencv.hpp>

using namespace cv;

void RgbToHsvSlowPixelAccess(const Mat &inputImage, Mat &outputImage);
void RgbToHsvEfficientPixelAccess(Mat &inputImage, Mat &outputImage);
