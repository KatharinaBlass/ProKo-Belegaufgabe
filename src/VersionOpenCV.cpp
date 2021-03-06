#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;
using namespace std;

int VersionOpenCV(int argc, char **argv)
{
  // read the image
  Mat image = imread(argv[2], IMREAD_COLOR);

  if (!image.data)
  {
    printf("No image data \n");
    return -1;
  };

  Mat cv_hsv_image;
  Mat cv_hsv_emboss_image;
  Mat cv_grayscale_image;

  // start time
  double t0 = omp_get_wtime();

  // OpenCv hsv conversion
  cvtColor(image, cv_hsv_image, COLOR_BGR2HSV);

  // OpenCV emboss filter
  float emboss_data[9] = {2, -0, 0, 0, -1, 0, 0, 0, -1};
  Mat embossKernel = Mat(3, 3, CV_32F, emboss_data);
  filter2D(cv_hsv_image, cv_hsv_emboss_image, -1, embossKernel, Point(-1, -1), 0, BORDER_DEFAULT);

  // OpenCV grayscale
  cvtColor(image, cv_grayscale_image, COLOR_BGR2GRAY);

  // end time
  double t1 = omp_get_wtime();
  cout << "Image Conversion took " << (t1 - t0) << " seconds" << endl;

  // save images as png files
  imwrite("image_grayscale.png", cv_grayscale_image);
  imwrite("image_hsv_emboss.png", cv_hsv_emboss_image);

  // display images and wait for a key-press, then close the window
  imshow("Original image", image);
  imshow("HSV and Emboss image", cv_hsv_emboss_image);
  imshow("Grayscale image", cv_grayscale_image);

  waitKey(0);
  destroyAllWindows();

  return 0;
}