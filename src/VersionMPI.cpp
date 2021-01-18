#include <opencv2/opencv.hpp>
#include <mpi.h>

#include "./RgbToGrayscale.hpp"
#include "./RgbToHsv.hpp"
#include "./EmbossFilter.hpp"

using namespace cv;
using namespace std;

int VersionMPI(int argc, char **argv, bool withOmp)
{
  int rank, size;

  // the full original image and the full converted grayscale and hsv/emboss images
  Mat full_image_original;
  Mat full_image_hsv_emboss;
  Mat full_image_grayscale;

  // image properties
  int image_properties[4];

  // init MPI
  MPI_Init(&argc, &argv);

  // get the size and rank
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // load the image ONLY in the master process #0
  if (rank == 0)
  {
    full_image_original = imread(argv[2], IMREAD_COLOR);
    if (!full_image_original.data)
    {
      printf("No image data \n");
      return -1;
    };

    // get the properties of the image, to send to other processes later
    // split the height by size so that each process will get its own horizonatal slice of the image later
    image_properties[0] = full_image_original.cols;        // width
    image_properties[1] = full_image_original.rows / size; // height, divided by number of processes
    image_properties[2] = full_image_original.type();      // image type (in this case: CV_8UC3)
    image_properties[3] = full_image_original.channels();  // number of channels (here: 3)

    // intialize result images
    full_image_grayscale = Mat::zeros(full_image_original.rows, image_properties[0], CV_8UC1);
    full_image_hsv_emboss = Mat::zeros(full_image_original.rows, image_properties[0], image_properties[2]);
  }

  // wait for it to finish
  MPI_Barrier(MPI_COMM_WORLD);

  // start time
  double start = MPI_Wtime();

  // now broadcast the image properties from process #0 to all others
  MPI_Bcast(image_properties, 4, MPI_INT, 0, MPI_COMM_WORLD);

  // now that all processes have these properties, initialize the "partial" images in each process
  Mat part_image_original = Mat(image_properties[1], image_properties[0], image_properties[2]);
  Mat part_image_hsv = Mat(image_properties[1], image_properties[0], image_properties[2]);
  Mat part_image_emboss = Mat(image_properties[1], image_properties[0], image_properties[2]);
  Mat part_image_grayscale = Mat(image_properties[1], image_properties[0], CV_8UC1);

  // wait for all to finish
  MPI_Barrier(MPI_COMM_WORLD);

  // calculate the number of bytes to send: (Height * Width * Channels) for RGB image and grayscale image
  int send_size = image_properties[1] * image_properties[0] * image_properties[3];
  int send_size_grayscale_image = image_properties[1] * image_properties[0];

  // from process #0 scatter to all others
  MPI_Scatter(full_image_original.data, send_size, MPI_UNSIGNED_CHAR,
              part_image_original.data, send_size, MPI_UNSIGNED_CHAR,
              0, MPI_COMM_WORLD);
  // now all the PROCESSES have their own copy of the 'part_image_original' which contains a horizontal slice of the image

  if (withOmp)
  {
    // for combined MPI and OpenMP version use OpenMP-parallelized converter functions
    RgbToHsvParallel(part_image_original, part_image_hsv);
    applyParallelEmbossFilter(part_image_hsv, part_image_emboss);
    RgbToGrayscaleParallel(part_image_original, part_image_grayscale);
  }
  else
  {
    // for MPI version used non-parallelized converter functions
    RgbToHsvEfficientPixelAccess(part_image_original, part_image_hsv);
    applyEmbossFilterEfficientPixelAccess(part_image_hsv, part_image_emboss);
    RgbToGrayscaleEfficientPixelAccess(part_image_original, part_image_grayscale);
  }

  // wait for all to finish
  MPI_Barrier(MPI_COMM_WORLD);

  // MPI_Gather both resulted images back into process #0
  MPI_Gather(part_image_emboss.data, send_size, MPI_UNSIGNED_CHAR,
             full_image_hsv_emboss.data, send_size, MPI_UNSIGNED_CHAR,
             0, MPI_COMM_WORLD);

  MPI_Gather(part_image_grayscale.data, send_size_grayscale_image, MPI_UNSIGNED_CHAR,
             full_image_grayscale.data, send_size_grayscale_image, MPI_UNSIGNED_CHAR,
             0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    // end time
    double end = MPI_Wtime();
    cout << "Image Conversion took " << (end - start) * 1000 << " milliseconds" << endl;

    // save images as png files
    imwrite("image_grayscale.png", full_image_grayscale);
    imwrite("image_hsv_emboss.png", full_image_hsv_emboss);

    // display images and wait for a key-press, then close the window
    imshow("Original image", full_image_original);
    imshow("HSV and Emboss image", full_image_hsv_emboss);
    imshow("Grayscale image", full_image_grayscale);

    waitKey(0);
    destroyAllWindows();
  }

  // finalize MPI
  MPI_Finalize();

  return 0;
}