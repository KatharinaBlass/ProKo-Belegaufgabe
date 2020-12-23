#include "opencv2/opencv.hpp"
#include "RgbToHsv.hpp"
#include <omp.h>

using namespace cv;
using namespace std;

void RgbToHsvEfficientPixelAccess(Mat &inputImage, Mat &outputImage)
{
  // check that the input image has at least three channels:
  if (inputImage.channels() < 3)
  {
    throw("Image doesn't have enough channels!");
  }

  // prepare an output 8-bit image of same size with three channels (for H, S and V)
  outputImage = Mat::zeros(inputImage.size(), CV_8UC3);

  for (int i = 0; i < inputImage.rows; i++)
  {
    Vec3b *inputImagePointer = inputImage.ptr<Vec3b>(i);
    Vec3b *outputImagePointer = outputImage.ptr<Vec3b>(i);

    for (int j = 0; j < inputImage.cols; j++)
    {
      // get normalized R, G and B values:
      double R = static_cast<double>(inputImagePointer[j][2]) / 255.0;
      double G = static_cast<double>(inputImagePointer[j][1]) / 255.0;
      double B = static_cast<double>(inputImagePointer[j][0]) / 255.0;

      // compute cmin, cmax and their difference
      double cmin = min({R, G, B});
      double cmax = max({R, G, B});
      double diff = cmax - cmin;

      // if cmin == max
      if (cmin == cmax)
      {
        outputImagePointer[j][0] = 0;
      }

      // if cmax == R, G or B
      if (cmax == R)
      {
        outputImagePointer[j][0] = static_cast<uchar>(static_cast<int>(60 * ((G - B) / diff) + 360.0) % 360);
      }
      else if (cmax == G)
      {
        outputImagePointer[j][0] = static_cast<uchar>(static_cast<int>(60 * ((B - R) / diff) + 120.0) % 360);
      }
      else if (cmax == B)
      {
        outputImagePointer[j][0] = static_cast<uchar>(static_cast<int>(60 * ((R - G) / diff) + 240.0) % 360);
      }

      // calculate S
      if (cmax == 0)
      {
        outputImagePointer[j][1] = 0;
      }
      else
      {
        outputImagePointer[j][1] = static_cast<uchar>((diff / cmax) * 100);
      }

      // calculate V
      outputImagePointer[j][2] = static_cast<uchar>(cmax * 255.0);
    }
  }
}

void RgbToHsvSlowPixelAccess(const Mat &inputImage, Mat &outputImage)
{
  // check that the input image has at least three channels:
  if (inputImage.channels() < 3)
  {
    throw("Image doesn't have enough channels!");
  }



  // prepare an output 8-bit image of same size with three channels (for H, S and V)
  // outputImage = Mat::zeros(inputImage.size(), CV_8UC3);

  outputImage = inputImage;

  Vec3b bgr_pixel;
  Vec3b hsv_pixel;

  for (int i = 0; i < inputImage.rows; i++)
  {
    for (int j = 0; j < inputImage.cols; j++)
    {
      // get BGR pixel:
      bgr_pixel = inputImage.at<Vec3b>(i, j);

      // get normalized R, G and B values:
      double R = static_cast<double>(bgr_pixel[2]) / 255.0;
      double G = static_cast<double>(bgr_pixel[1]) / 255.0;
      double B = static_cast<double>(bgr_pixel[0]) / 255.0;

      // compute cmin, cmax and their difference
      double cmin = min({R, G, B});
      double cmax = max({R, G, B});
      double diff = cmax - cmin;

      // if cmin == max
      if (cmin == cmax)
      {
        hsv_pixel[0] = 0;
      }

      // if cmax == R, G or B
      if (cmax == R)
      {
        hsv_pixel[0] = static_cast<uchar>(static_cast<int>(60 * ((G - B) / diff) + 360.0) % 360);
      }
      else if (cmax == G)
      {
        hsv_pixel[0] = static_cast<uchar>(static_cast<int>(60 * ((B - R) / diff) + 120.0) % 360);
      }
      else if (cmax == B)
      {
        hsv_pixel[0] = static_cast<uchar>(static_cast<int>(60 * ((R - G) / diff) + 240.0) % 360);
      }

      // calculate S
      if (cmax == 0)
      {
        hsv_pixel[1] = 0;
      }
      else
      {
        hsv_pixel[1] = static_cast<uchar>((diff / cmax) * 100);
      }

      // calculate V
      hsv_pixel[2] = static_cast<uchar>(cmax * 255.0);

      outputImage.at<Vec3b>(i, j) = hsv_pixel;
    }
  }
}


void RgbToHsvParallel(Mat &inputImage, Mat &outputImage)
{
  // check that the input image has at least three channels:
  if (inputImage.channels() < 3)
  {
    throw("Image doesn't have enough channels!");
  }

  // prepare an output 8-bit image of same size with three channels (for H, S and V)
  outputImage = Mat::zeros(inputImage.size(), CV_8UC3);

  #pragma omp parallel for
  for (int i = 0; i < inputImage.rows; i++)
  {
    Vec3b *inputImagePointer = inputImage.ptr<Vec3b>(i);
    Vec3b *outputImagePointer = outputImage.ptr<Vec3b>(i);

    for (int j = 0; j < inputImage.cols; j++)
    {
      // get normalized R, G and B values:
      double R = static_cast<double>(inputImagePointer[j][2]) / 255.0;
      double G = static_cast<double>(inputImagePointer[j][1]) / 255.0;
      double B = static_cast<double>(inputImagePointer[j][0]) / 255.0;

      // compute cmin, cmax and their difference
      double cmin = min({R, G, B});
      double cmax = max({R, G, B});
      double diff = cmax - cmin;

      // if cmin == max
      if (cmin == cmax)
      {
        outputImagePointer[j][0] = 0;
      }

      // if cmax == R, G or B
      if (cmax == R)
      {
        outputImagePointer[j][0] = static_cast<uchar>(static_cast<int>(60 * ((G - B) / diff) + 360.0) % 360);
      }
      else if (cmax == G)
      {
        outputImagePointer[j][0] = static_cast<uchar>(static_cast<int>(60 * ((B - R) / diff) + 120.0) % 360);
      }
      else if (cmax == B)
      {
        outputImagePointer[j][0] = static_cast<uchar>(static_cast<int>(60 * ((R - G) / diff) + 240.0) % 360);
      }

      // calculate S
      if (cmax == 0)
      {
        outputImagePointer[j][1] = 0;
      }
      else
      {
        outputImagePointer[j][1] = static_cast<uchar>((diff / cmax) * 100);
      }

      // calculate V
      outputImagePointer[j][2] = static_cast<uchar>(cmax * 255.0);
    }
  }
}
