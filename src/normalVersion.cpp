#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "./RgbToGrayscale.hpp"
#include "./RgbToHsv.hpp"
#include "./EmbossFilter.hpp"

using namespace cv;
using namespace std;

// without any parallelization
int normalVersion(int argc, char **argv, bool slow = false)
{
    Mat image = imread(argv[2], IMREAD_COLOR);

    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    };

    Mat hsv_image;
    Mat hsv_emboss_image;
    Mat gray_image;

    double t0 = omp_get_wtime(); // start time

    if (slow)
    {
        RgbToHsvSlowPixelAccess(image, hsv_image);
        applyEmbossFilterSlowPixelAccess(image, hsv_emboss_image);
        RgbToGrayscaleSlowPixelAccess(image, gray_image);
    }
    else
    {
        RgbToHsvEfficientPixelAccess(image, hsv_image);
        applyEmbossFilterEfficientPixelAccess(image, hsv_emboss_image);
        RgbToGrayscaleEfficientPixelAccess(image, gray_image);
    }

    double t1 = omp_get_wtime(); // end time
    std::cout << "Image Conversion took " << (t1 - t0) * 1000 << " milliseconds" << std::endl;

    // save images as png files
    cv::imwrite("image_grayscale.png", gray_image);
    cv::imwrite("image_hsv_emboss.png", hsv_emboss_image);

    // display images and wait for a key-press, then close the window
    imshow("Original image", image);
    imshow("HSV and Emboss image", hsv_emboss_image);
    imshow("Grayscale image", gray_image);

    waitKey(0);
    destroyAllWindows();

    return 0;
}