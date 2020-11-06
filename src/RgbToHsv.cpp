#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "RgbToHsv.hpp"

// ---------------------------------------------------------------------------------------------------------------------------
// a C-style macro for getting the pixel [i,j] quickly, returns an array [B, G, R(, A)]
#define PIXEL(image, i, j) ((uchar*)image.data + image.channels() * (static_cast<int>(i) * image.cols + static_cast<int>(j)))

// ---------------------------------------------------------------------------------------------------------------------------

// Function to transform an RGB image into the HSV colour space
void RgbToHsv( const cv::Mat& inputImage, cv::Mat& outputImage ) 
{
  // check that the input image has at least three channels:
  if ( inputImage.channels() < 3 ) {
    throw("Image doesn't have enough channels!");
  }

  // prepare an output 8-bit image of same size with three channels (for H, S and V)
  outputImage = cv::Mat::zeros( inputImage.size(), CV_8UC3 );

  uchar* bgr_pixel;
  uchar* hsv_pixel;

  for ( int i = 0; i < inputImage.rows; i++ )
  {
    for ( int j = 0; j < inputImage.cols; j++ )
    {
      // get BGR pixel:
      bgr_pixel = PIXEL( inputImage, i, j );

      // get HSV pixel:
      hsv_pixel = PIXEL( outputImage, i, j );

      // get normalized R, G and B values:
      double R = static_cast<double>(bgr_pixel[2]) / 255.0;
      double G = static_cast<double>(bgr_pixel[1]) / 255.0;
      double B = static_cast<double>(bgr_pixel[0]) / 255.0;

      // compute cmin, cmax and their difference
      double cmin = std::min( { R, G, B } );
      double cmax = std::max( { R, G, B } );
      double diff = cmax - cmin;

      // if cmin == max
      if ( cmin == cmax ) {
        hsv_pixel[0] = 0;
      }

      // if cmax == R, G or B
      if ( cmax == R ) {
        hsv_pixel[0] = static_cast<uchar>( static_cast<int>( 60 * ((G - B) / diff) + 360.0 ) % 360 );
      }
      else if ( cmax == G ) {
        hsv_pixel[0] = static_cast<uchar>(static_cast<int>(60 * ((B - R) / diff) + 120.0) % 360);
      }
      else if ( cmax == B ) {
        hsv_pixel[0] = static_cast<uchar>(static_cast<int>(60 * ((R - G) / diff) + 240.0) % 360);
      }

      // calculate S
      if ( cmax == 0 ) {
        hsv_pixel[1] = 0;
      }
      else {
        hsv_pixel[1] = static_cast<uchar>((diff / cmax) * 100);
      }

      // calculate V
      hsv_pixel[2] = static_cast<uchar>(cmax * 255.0);
    }
  }
}
