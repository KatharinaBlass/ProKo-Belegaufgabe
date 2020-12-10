#include <opencv2/opencv.hpp>

using namespace cv;

void applyEmbossFilterSlowPixelAccess(const Mat &inputImage, Mat &outputImage);
void applyEmbossFilterEfficientPixelAccess(Mat &inputImage, Mat &outputImage);
