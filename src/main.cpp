
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <mpi.h>
#include <omp.h>
#include "RgbToHsv.hpp"
#include "EmbossFilter.hpp"
#include "RgbToGrayscale.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    // openCV example
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    cv::Mat image = cv::imread( argv[1], cv::IMREAD_COLOR );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    Mat hsv_image;
    Mat emboss_image;
    Mat gray_image;

    // own solutions
    RgbToHsvSlowPixelAccess( image, hsv_image );
    applyEmbossFilterSlowPixelAccess(hsv_image, emboss_image);
    RgbToGrayscaleSlowPixelAccess( image, gray_image );

    // openCV solution for hsv
    Mat cv_hsv_image;
    cvtColor(image, cv_hsv_image, COLOR_RGB2HSV);

    // openCV solution for emboss filter
    Mat cv_emboss_image = cv::Mat::zeros( image.size(), image.type() );
    float emboss2_data[9] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
    float emboss_data[9] = { 2, -0, 0, 0, -1, 0, 0, 0, -1 };
    cv::Mat embossKernel = cv::Mat(3, 3, CV_32F, emboss_data);
    cv::filter2D(hsv_image, cv_emboss_image, -1, embossKernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // openCV solution for grayscale
    Mat cv_grayscale_image;
    cvtColor(image, cv_grayscale_image, CV_RGB2GRAY);

    // display images and wait for a key-press, then close the window
    cv::imshow( "RGB image", image );
    cv::imshow( "HSV image", hsv_image );
    cv::imshow( "OpenCV hsv image", cv_hsv_image);
    cv::imshow( "Emboss image", emboss_image);
    cv::imshow( "OpenCV emboss image", cv_emboss_image);
    cv::imshow( "Grayscale image", gray_image);
    cv::imshow( "OpenCV grayscale image", cv_grayscale_image);

    waitKey(0);
    destroyAllWindows();

    return 0;
}