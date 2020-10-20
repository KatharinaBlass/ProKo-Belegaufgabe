#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <omp.h>

using namespace cv;

int main(int argc, char** argv )
{
    // std::cout << "hello world!";
    /*if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], 1 );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    waitKey(0);*/

    #pragma omp parallel for
    for (int i=0; i<4; i++) {
        printf("hello aus dem theead %d von %d\n", omp_get_thread_num(), omp_get_num_threads());
    }

    return 0;
}