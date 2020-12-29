/**
  Sandbox for teaching the basics of MPI with C/C++
  for the "Programmierkonepte und Algorithmen" course
*/

#include <iostream>
#include <string>

#include <mpi.h>
#include <omp.h>

#include "RgbToHsv.hpp"
#include "EmbossFilter.hpp"
#include "RgbToGrayscale.hpp"
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
    int rank, size;

    // the full image:
    cv::Mat full_image_original;
    cv::Mat full_image_hsv_emboss;
    cv::Mat full_image_grayscale;

    // image properties:
    int image_properties[4];

    // init MPI
    MPI_Init(&argc, &argv);

    // get the size and rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // load the image ONLY in the master process #0:
    if (rank == 0)
    {
        full_image_original = cv::imread(argv[1], cv::IMREAD_COLOR);

        // get the properties of the image, to send to other processes later:
        image_properties[0] = full_image_original.cols;       // width
        image_properties[1] = full_image_original.rows;       // height
        image_properties[2] = full_image_original.type();     // image type (in this case: CV_8UC3)
        image_properties[3] = full_image_original.channels(); // number of channels (here: 3)

        //full_image_original.copyTo(full_image_hsv_emboss);
        full_image_grayscale = cv::Mat::zeros(full_image_original.rows, full_image_original.cols, CV_8UC1);
        full_image_hsv_emboss = cv::Mat::zeros(full_image_original.rows, full_image_original.cols, full_image_original.type());
    }

    for (int i = 1; i <= 20; i++)
    {
        double t0 = omp_get_wtime(); // start time

        // now broadcast the image properties from process #0 to all others:
        // the 'image_properties' array is only initialized in process #0!
        // that's why your IDE might show a warning here
        MPI_Bcast(image_properties, 4, MPI_INT, 0, MPI_COMM_WORLD);

        // now all processes have these properties, initialize the "partial" image in each process
        cv::Mat local_image_original = cv::Mat(image_properties[1], image_properties[0], image_properties[2]);
        cv::Mat local_image_hsv = cv::Mat(image_properties[1], image_properties[0], image_properties[2]);
        cv::Mat local_image_emboss = cv::Mat(image_properties[1], image_properties[0], image_properties[2]);
        cv::Mat local_image_grayscale = cv::Mat(image_properties[1], image_properties[0], CV_8UC1);

        // wait for all to finish:
        MPI_Barrier(MPI_COMM_WORLD);

        // the number of bytes to send: (Height * Width * Channels)
        int send_size = image_properties[1] * image_properties[0] * image_properties[3];
        int send_size_grayscale_image = image_properties[1] * image_properties[0];

        if (rank == 0)
        {
            full_image_original.copyTo(local_image_original);
        }

        // broadcast the image data to each process
        MPI_Bcast(local_image_original.data, send_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        // now all the PROCESSES have their own copy of the 'local_image_original'
        // we can do something with it...

        if (rank == 1)
        {
            RgbToHsvEfficientPixelAccess(local_image_original, local_image_hsv);
            applyEmbossFilterEfficientPixelAccess(local_image_hsv, local_image_emboss);
            //send it back into process #0:
            MPI_Send(local_image_emboss.data, send_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        else if (rank == 2)
        {
            RgbToGrayscaleEfficientPixelAccess(local_image_original, local_image_grayscale);
            //send it back into process #0:
            MPI_Send(local_image_grayscale.data, send_size_grayscale_image, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        else if (rank == 0)
        {
            // receive the modified images
            MPI_Recv(full_image_hsv_emboss.data, send_size, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(full_image_grayscale.data, send_size_grayscale_image, MPI_UNSIGNED_CHAR, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            double t1 = omp_get_wtime(); // end time
            std::cout << "Processing took " << (t1 - t0) << " seconds" << std::endl;
            //std::cout << "Process #0 received the gathered image" << std::endl;

            /*cv::imshow("hsv emboss image", full_image_hsv_emboss);
            cv::imshow("grayscale image", full_image_grayscale);
            cv::imshow("original image", full_image_original);

            cv::waitKey(0); // will need to press a key in EACH process...
            cv::destroyAllWindows();*/
        }
    }
    // finalize MPI
    MPI_Finalize();
}