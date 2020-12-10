#include <opencv2/opencv.hpp>

using namespace cv;

void RgbToGrayscaleSlowPixelAccess(const Mat &inputImage, Mat &outputImage);
void RgbToGrayscaleEfficientPixelAccess(Mat &inputImage, Mat &outputImage);
