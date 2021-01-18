#include <opencv2/opencv.hpp>
#include <omp.h>

#include "./RgbToGrayscale.hpp"
#include "./RgbToHsv.hpp"
#include "./EmbossFilter.hpp"

using namespace cv;
using namespace std;

int VersionOpenMP(int argc, char **argv)
{
    // read the image
    Mat image = imread(argv[2], IMREAD_COLOR);

    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    };

    Mat hsv_image;
    Mat hsv_emboss_image;
    Mat gray_image;

    // start time
    double t0 = omp_get_wtime();

    // use OpenMP-parallelized conversion functions
    RgbToHsvParallel(image, hsv_image);
    applyParallelEmbossFilter(hsv_image, hsv_emboss_image);
    RgbToGrayscaleParallel(image, gray_image);

    // end time
    double t1 = omp_get_wtime();
    cout << "Image Conversion took " << (t1 - t0) * 1000 << " milliseconds" << endl;

    // save images as png files
    imwrite("image_grayscale.png", gray_image);
    imwrite("image_hsv_emboss.png", hsv_emboss_image);

    // display images and wait for a key-press, then close the window
    imshow("Original image", image);
    imshow("HSV and Emboss image", hsv_emboss_image);
    imshow("Grayscale image", gray_image);

    waitKey(0);
    destroyAllWindows();

    return 0;
}