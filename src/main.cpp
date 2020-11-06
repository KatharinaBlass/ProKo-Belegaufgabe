
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <mpi.h>
#include <omp.h>
#include "RgbToHsv.hpp"
#include "EmbossFilter.hpp"

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

    cv::Mat image = cv::imread( argv[1], cv::IMREAD_UNCHANGED );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    Mat hsv_image;
    Mat emboss_image;

    RgbToHsv( image, hsv_image );
    applyEmbossFilter(hsv_image, emboss_image, true);

    // openCv solution
    Mat outputImage = cv::Mat::zeros( image.size(), image.type() );
    float emboss2_data[9] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
    float emboss_data[9] = { 2, -0, 0, 0, -1, 0, 0, 0, -1 };
    cv::Mat embossKernel = cv::Mat(3, 3, CV_32F, emboss_data);

    cv::filter2D(hsv_image, outputImage, -1, embossKernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    // display and wait for a key-press, then close the window
    cv::imshow( "RGB image", image );
    cv::imshow( "HSV image", hsv_image );
    cv::imshow( "Emboss image", emboss_image);
    cv::imshow( "OpenCV emboss image", outputImage);

    waitKey(0);
    destroyAllWindows();

    return 0;
}