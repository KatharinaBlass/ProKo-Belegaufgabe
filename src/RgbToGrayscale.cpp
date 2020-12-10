#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "RgbToGrayscale.hpp"

void RgbToGrayscale(const cv::Mat &inputImage, cv::Mat &outputImage)
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
  outputImage = cv::Mat::zeros(inputImage.size(), CV_8UC1);

  cv::Vec3b bgr_pixel;
  int Gray;

  for (int i = 0; i < inputImage.rows; i++)
  {
    for (int j = 0; j < inputImage.cols; j++)
    {
      // get BGR pixel:
      bgr_pixel = inputImage.at<cv::Vec3b>(i, j);

      // get R, G and B values:
      double R = static_cast<double>(bgr_pixel[2]);
      double G = static_cast<double>(bgr_pixel[1]);
      double B = static_cast<double>(bgr_pixel[0]);

      // calculate Gray value:
      Gray = 0.21 * R + 0.72 * G + 0.07 * B;

      // set gray pixel:
      outputImage.at<uchar>(i, j) = Gray;
    }
  }
}