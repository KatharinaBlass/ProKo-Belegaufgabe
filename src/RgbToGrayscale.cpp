#include "opencv2/opencv.hpp"
#include "RgbToGrayscale.hpp"
#include <omp.h>

using namespace cv;

void RgbToGrayscaleEfficientPixelAccess(Mat &inputImage, Mat &outputImage)
{
  // check that the input image has at least three channels:
  if (inputImage.channels() < 3)
  {
    throw("Image doesn't have enough channels!");
  }

  // check that the output image has a single channel:
  if (outputImage.channels() > 1)
  {
    throw("Image has to much channels!");
  }

  // prepare an output image of same size with 1 channel
  outputImage = Mat::zeros(inputImage.size(), CV_8UC1);

  int grayPixelValue;

  for (int i = 0; i < inputImage.rows; i++)
  {
    // We obtain a pointer to the beginning of row i of inputImage and another one for outputImage
    Vec3b *inputImagePointer = inputImage.ptr<Vec3b>(i);
    uchar *outputImagePointer = outputImage.ptr<uchar>(i);

    for (int j = 0; j < inputImage.cols; j++)
    {
      // get R, G and B values:
      double R = static_cast<double>(inputImagePointer[j][2]);
      double G = static_cast<double>(inputImagePointer[j][1]);
      double B = static_cast<double>(inputImagePointer[j][0]);

      // calculate grayPixelValue:
      grayPixelValue = 0.21 * R + 0.72 * G + 0.07 * B;

      // set grayPixelValue to outputImagePointer:
      outputImagePointer[j] = grayPixelValue;
    }
  }
}

void RgbToGrayscaleSlowPixelAccess(const Mat &inputImage, Mat &outputImage)
{
  // check that the input image has at least three channels:
  if (inputImage.channels() < 3)
  {
    throw("Image doesn't have enough channels!");
  }

  // check that the output image has a single channel:
  if (outputImage.channels() > 1)
  {
    throw("Image has to much channels!");
  }

  // prepare an output image of same size with 1 channel
  outputImage = Mat::zeros(inputImage.size(), CV_8UC1);

  Vec3b bgr_pixel;
  int grayPixelValue;

  for (int i = 0; i < inputImage.rows; i++)
  {
    for (int j = 0; j < inputImage.cols; j++)
    {
      // get BGR pixel:
      bgr_pixel = inputImage.at<Vec3b>(i, j);

      // get R, G and B values:
      double R = static_cast<double>(bgr_pixel[2]);
      double G = static_cast<double>(bgr_pixel[1]);
      double B = static_cast<double>(bgr_pixel[0]);

      // calculate grayPixelValue:
      grayPixelValue = 0.21 * R + 0.72 * G + 0.07 * B;

      // set grayPixelValue:
      outputImage.at<uchar>(i, j) = grayPixelValue;
    }
  }
}

void RgbToGrayscaleParallel(Mat &inputImage, Mat &outputImage)
{
  // check that the input image has at least three channels:
  if (inputImage.channels() < 3)
  {
    throw("Image doesn't have enough channels!");
  }

  // check that the output image has a single channel:
  if (outputImage.channels() > 1)
  {
    throw("Image has to much channels!");
  }

  // prepare an output image of same size with 1 channel
  outputImage = Mat::zeros(inputImage.size(), CV_8UC1);

  int grayPixelValue;

#pragma omp parallel for
  for (int i = 0; i < inputImage.rows; i++)
  {
    // We obtain a pointer to the beginning of row i of inputImage and another one for outputImage
    Vec3b *inputImagePointer = inputImage.ptr<Vec3b>(i);
    uchar *outputImagePointer = outputImage.ptr<uchar>(i);

    for (int j = 0; j < inputImage.cols; j++)
    {
      // get R, G and B values:
      double R = static_cast<double>(inputImagePointer[j][2]);
      double G = static_cast<double>(inputImagePointer[j][1]);
      double B = static_cast<double>(inputImagePointer[j][0]);

      // calculate grayPixelValue:
      grayPixelValue = 0.21 * R + 0.72 * G + 0.07 * B;

      // set grayPixelValue to outputImagePointer:
      outputImagePointer[j] = grayPixelValue;
    }
  }
}
