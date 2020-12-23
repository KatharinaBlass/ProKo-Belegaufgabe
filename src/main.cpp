/**
  Sandbox for teaching the basics of MPI with C/C++
  for the "Programmierkonepte und Algorithmen" course
*/

#include <iostream>
#include <string>

#include <mpi.h>

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
        full_image_original = cv::imread("../lenna.jpg", cv::IMREAD_COLOR);

        // get the properties of the image, to send to other processes later:
        image_properties[0] = full_image_original.cols;        // width
        image_properties[1] = full_image_original.rows / size; // height, divide it by number of processes
        image_properties[2] = full_image_original.type();      // image type (in this case: CV_8UC3)
        image_properties[3] = full_image_original.channels();  // number of channels (here: 3)

        full_image_original.copyTo(full_image_hsv_emboss);
        full_image_grayscale = cv::Mat::zeros(full_image_original.rows, full_image_original.cols, CV_8UC1);
    }

    // wait for it to finish:
    MPI_Barrier(MPI_COMM_WORLD);

    // now broadcast the image properties from process #0 to all others:
    // the 'image_properties' array is only initialized in process #0!
    // that's why your IDE might show a warning here
    MPI_Bcast(image_properties, 4, MPI_INT, 0, MPI_COMM_WORLD);

    // now all processes have these properties, initialize the "partial" image in each process
    cv::Mat part_image_original = cv::Mat(image_properties[1], image_properties[0], image_properties[2]);
    cv::Mat part_image_hsv = cv::Mat(image_properties[1], image_properties[0], image_properties[2]);
    cv::Mat part_image_emboss = cv::Mat(image_properties[1], image_properties[0], image_properties[2]);
    cv::Mat part_image_grayscale = cv::Mat(image_properties[1], image_properties[0], CV_8UC1);

    // wait for all to finish:
    MPI_Barrier(MPI_COMM_WORLD);

    // now we can cut the full_image into parts and send the parts to each process!

    // first, the number of bytes to send: (Height * Width * Channels)
    int send_size = image_properties[1] * image_properties[0] * image_properties[3];
    int send_size_grayscale_image = image_properties[1] * image_properties[0];

    // from process #0 scatter to all others:
    // the cv::Mat.data is a pointer to the raw image data (B1,G1,R1,B2,G2,R2,.....)
    MPI_Scatter(full_image_original.data, send_size, MPI_UNSIGNED_CHAR, // unsigned char = unsigned 8-bit (byte)
                part_image_original.data, send_size, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD); // from process #0

    // of course, you can Bcast the image instead of scattering it

    // now all the PROCESSES have their own copy of the 'part_image' image, which contains
    // a horizontal slice of the image
    // we can do something with it...

    RgbToHsvEfficientPixelAccess(part_image_original, part_image_hsv);
    applyEmbossFilterEfficientPixelAccess(part_image_hsv, part_image_emboss);
    RgbToGrayscaleEfficientPixelAccess(part_image_original, part_image_grayscale);

    // wait for all to finish:
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI_Gather it back into process #0:
    MPI_Gather(part_image_emboss.data, send_size, MPI_UNSIGNED_CHAR,
               full_image_hsv_emboss.data, send_size, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    MPI_Gather(part_image_grayscale.data, send_size_grayscale_image, MPI_UNSIGNED_CHAR,
               full_image_grayscale.data, send_size_grayscale_image, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Process #0 received the gathered image" << std::endl;

        cv::imshow("hsv emboss image", full_image_hsv_emboss);
        cv::imshow("grayscale image", full_image_grayscale);
        cv::imshow("original image", full_image_original);

        cv::waitKey(0); // will need to press a key in EACH process...
        cv::destroyAllWindows();
    }

    // finalize MPI
    MPI_Finalize();
}