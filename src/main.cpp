#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <mpi.h>
#include <omp.h>

using namespace cv;

int main(int argc, char** argv )
{
    /* C++ example
    std::cout << "hello world!";
    */

    /* openCV example
    if ( argc != 2 )
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

    waitKey(0);
    */

    /* openMP example
    #pragma omp parallel for
    for (int i=0; i<4; i++) {
        printf("hello aus dem theead %d von %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    */

    // MPI example
    MPI_Init(&argc,&argv);
    //parallel code

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    char name[128]; int name_len;
    MPI_Get_processor_name( name, &name_len );

    printf("hello from < %d > of < %d > in %d\n",rank, size, name);

    MPI_Finalize();

    return 0;
}