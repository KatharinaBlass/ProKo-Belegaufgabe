
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

int main(int argc, char **argv)
{
    // openCV example
    if (argc != 2)
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image = imread(argv[1], IMREAD_COLOR);

    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    Mat hsv_image;
    Mat emboss_image;
    Mat gray_image;

    for (int i = 1; i <= 20; i++)
    {
        double t0 = omp_get_wtime(); // start time
        // own solutions
/*
        RgbToHsvSlowPixelAccess( image, hsv_image );
        applyEmbossFilterSlowPixelAccess(hsv_image, emboss_image);
        RgbToGrayscaleSlowPixelAccess( image, gray_image );
*/
        RgbToHsvParallel(image, hsv_image);
        applyParallelEmbossFilter(hsv_image, emboss_image);
        RgbToGrayscaleParallel(image, gray_image);

        double t1 = omp_get_wtime(); // end time
        std::cout << "Processing took " << (t1 - t0) << " seconds" << std::endl;
    }
/*
    // display images and wait for a key-press, then close the window
    imshow("RGB image", image);
    imshow("HSV image", hsv_image);
    imshow("Emboss image", emboss_image);
    imshow("Grayscale image", gray_image);

    waitKey(0);
    destroyAllWindows();*/

    return 0;
}