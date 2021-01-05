#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <mpi.h>

#include "./RgbToGrayscale.hpp"
#include "./RgbToHsv.hpp"
#include "./EmbossFilter.hpp"

int mpiVersion(int argc, char **argv)
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
    full_image_original = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (!full_image_original.data)
    {
      printf("No image data \n");
      return -1;
    };

    // get the properties of the image, to send to other processes later:
    image_properties[0] = full_image_original.cols;       // width
    image_properties[1] = full_image_original.rows;       // height
    image_properties[2] = full_image_original.type();     // image type (in this case: CV_8UC3)
    image_properties[3] = full_image_original.channels(); // number of channels (here: 3)

    //full_image_original.copyTo(full_image_hsv_emboss);
    full_image_grayscale = cv::Mat::zeros(image_properties[1], image_properties[0], CV_8UC1);
    full_image_hsv_emboss = cv::Mat::zeros(image_properties[1], image_properties[0], image_properties[2]);
  }

  double start = MPI_Wtime();

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

    // only MPI
    RgbToHsvEfficientPixelAccess(local_image_original, local_image_hsv);
    applyEmbossFilterEfficientPixelAccess(local_image_hsv, local_image_emboss);

    //send it back into process #0:
    MPI_Send(local_image_emboss.data, send_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
  }
  else if (rank == 2)
  {

    // only MPI
    RgbToGrayscaleEfficientPixelAccess(local_image_original, local_image_grayscale);

    //send it back into process #0:
    MPI_Send(local_image_grayscale.data, send_size_grayscale_image, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
  }
  else if (rank == 0)
  {
    // receive the modified images
    MPI_Recv(full_image_hsv_emboss.data, send_size, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(full_image_grayscale.data, send_size_grayscale_image, MPI_UNSIGNED_CHAR, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    double end = MPI_Wtime();
    std::cout << "Image Conversion took " << (end - start) << " seconds" << std::endl;

    // save images as png files
    cv::imwrite("image_grayscale.png", full_image_grayscale);
    cv::imwrite("image_hsv_emboss.png", full_image_hsv_emboss);

    // display images and wait for a key-press, then close the window
    cv::imshow("Original image", full_image_original);
    cv::imshow("HSV and Emboss image", full_image_hsv_emboss);
    cv::imshow("Grayscale image", full_image_grayscale);

    cv::waitKey(0); // will need to press a key in EACH process...
    cv::destroyAllWindows();
  }

  // finalize MPI
  MPI_Finalize();

  return 0;
}