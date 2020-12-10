#include "opencv2/opencv.hpp"
#include "EmbossFilter.hpp"

using namespace cv;

uchar convolutePixelEfficient(Mat &inputImage, const Mat &kernel, int I, int J, int K)
{
  int kSize = kernel.rows;
  int halfSize = kSize / 2;

  double newPixelValue = 0;

  for (int i = 0; i < kSize; i++)
  {
    for (int j = 0; j < kSize; j++)
    {
      double inputImagePixelValue = static_cast<double>(inputImage.ptr<Vec3b>(I + i - halfSize)[J + j - halfSize][K]);
      newPixelValue += inputImagePixelValue * kernel.at<float>(i, j);
    }
  }
  return static_cast<uchar>(min(255, max(0, int(round(newPixelValue)))));
}

void applyEmbossFilterEfficientPixelAccess(Mat &inputImage, Mat &outputImage)
{
  // create a kernel
  float emboss_data[9] = {2, -0, 0, 0, -1, 0, 0, 0, -1};
  Mat embossKernel = Mat(3, 3, CV_32F, emboss_data);

  // initialize the empty output image:
  outputImage = Mat::zeros(inputImage.size(), inputImage.type());

  // go over the image:
  for (int i = 1; i < inputImage.rows - 1; i++)
  {
    // We obtain a pointer to the beginning of row i of inputImage and another one for outputImage
    Vec3b *outputImagePointer = outputImage.ptr<Vec3b>(i);

    for (int j = 1; j < inputImage.cols - 1; j++)
    {
      for (int k = 0; k < inputImage.channels(); k++)
      {
        outputImagePointer[j][k] = convolutePixelEfficient(inputImage, embossKernel, i, j, k);
      }
    }
  }
}

/**
  Perform convolution on the [I, J] pixel in the K channel of the input image with the given kernel,
  as a result returns the sum of the multiplied values, e.g. for a Kernel of size [3 x 3]:
  img[I-1,J-1,K] * Kernel[0,0] + img[I-1,J,K] * Kernel[0,1] + ... + img[I+1,J,K] * Kernel[2,1] + img[I+1,J+1,K] * Kernel[2,2]
*/
uchar convolutePixel(const Mat &inputImage, const Mat &kernel, int I, int J, int K)
{
  int kSize = kernel.rows;
  int halfSize = kSize / 2;

  double pixelValue = 0;

  for (int i = 0; i < kSize; i++)
  {
    for (int j = 0; j < kSize; j++)
    {
      uchar inputImagePixel = inputImage.at<Vec3b>(I + i - halfSize, J + j - halfSize)[K];
      pixelValue += static_cast<double>(inputImagePixel) * kernel.at<float>(i, j);
    }
  }
  return static_cast<uchar>(min(255, max(0, int(round(pixelValue)))));
}

/**
  Function to apply an Emboss filter on the input image, with an Emboss Kernel
  The result will be written into the outputImage (the image will be overwritten, if outputImage is not empty)
  Setting the 'padImage' flag to TRUE will pad the input image on all sides with zeros, to properly deal
  with the image edges
*/
void applyEmbossFilterSlowPixelAccess(const Mat &inputImage, Mat &outputImage)
{
  // create a kernel
  float emboss_data[9] = {2, -0, 0, 0, -1, 0, 0, 0, -1};
  Mat embossKernel = Mat(3, 3, CV_32F, emboss_data);

  // initialize the empty output image:
  outputImage = Mat::zeros(inputImage.size(), inputImage.type());

  // go over the image:
  for (int i = 1; i < inputImage.rows - 1; i++)
  {
    for (int j = 1; j < inputImage.cols - 1; j++)
    {
      for (int k = 0; k < inputImage.channels(); k++)
      {
        outputImage.at<Vec3b>(i, j)[k] = convolutePixel(inputImage, embossKernel, i, j, k);
      }
    }
  }
}
