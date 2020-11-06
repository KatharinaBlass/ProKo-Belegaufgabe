#include <iostream>
#include <cstdio>
#include <string>
#include <stdio.h>
#include <omp.h>
#include "opencv2/opencv.hpp"
#include <cmath>
#include "EmbossFilter.hpp"

#define _USE_MATH_DEFINES

using namespace cv;
using namespace std;

// a C-style macro for getting the pixel [i,j] quickly, returns an array [B, G, R(, A)]
#define PIXEL(image, i, j) ((uchar*)image.data + image.channels() * (static_cast<int>(i) * image.cols + static_cast<int>(j)))

/**
  Perform convolution on the [I, J] pixel in the K channel of the input image with the given kernel,
  as a result returns the sum of the multiplied values, e.g. for a Kernel of size [3 x 3]:
  img[I-1,J-1,K] * Kernel[0,0] + img[I-1,J,K] * Kernel[0,1] + ... + img[I+1,J,K] * Kernel[2,1] + img[I+1,J+1,K] * Kernel[2,2]
*/
uchar convolutePixel( const Mat& inputImage, const Mat& kernel, int I, int J, int K ) {
  int kSize = kernel.rows;
  int halfSize = kSize / 2;

  double pixelValue = 0;

  for ( int i = 0; i < kSize; i++ )
  {
    for ( int j = 0; j < kSize; j++ )
    {
      auto pixel = PIXEL( inputImage, I + i - halfSize, J + j - halfSize );
      pixelValue += static_cast<double>(pixel[K]) * kernel.at<float>(i,j);
    }
  }
  return static_cast<uchar>( min( 255, max( 0, int(round( pixelValue )) )));
}

/**
  Function to apply an Emboss filter on the input image, with an Emboss Kernel
  The result will be written into the outputImage (the image will be overwritten, if outputImage is not empty)
  Setting the 'padImage' flag to TRUE will pad the input image on all sides with zeros, to properly deal
  with the image edges
*/
void applyEmbossFilter( const Mat& inputImage, Mat& outputImage, bool padImage )
{
  // create a kernel
  float emboss_data[9] = { 2, -0, 0, 0, -1, 0, 0, 0, -1 };
  float emboss2_data[9] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
  Mat embossKernel = Mat(3, 3, CV_32F, emboss_data);

  // pad image with zeros on all sides (optional step):
  Mat paddedImage;
  if ( padImage ) {
    paddedImage = Mat::zeros( inputImage.rows + 2, inputImage.cols + 2, inputImage.type() );
    inputImage.copyTo( paddedImage( Rect( 1, 1, inputImage.cols, inputImage.rows ) ) );
  }
  else {
    paddedImage = inputImage.clone();
  }

  // initialize the empty output image:
  outputImage = Mat::zeros( paddedImage.size(), paddedImage.type() );

  // go over the image:
  uchar* pixel;
  for ( int i = 1; i < paddedImage.rows - 1; i++ )
  {
    for ( int j = 1; j < paddedImage.cols - 1; j++ )
    {
      pixel = PIXEL( outputImage, i, j );
      for ( int k = 0; k < paddedImage.channels(); k++ )
      {
        pixel[k] = convolutePixel( paddedImage, embossKernel, i, j, k );
      }
    }
  }

  // unpad the image (remove the zero-padding):
  if ( padImage ) {
    outputImage( Rect( 1, 1, inputImage.cols, inputImage.rows ) ).copyTo( outputImage );
  }
}
