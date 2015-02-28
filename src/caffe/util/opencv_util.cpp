#include <string>

#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include "caffe/util/opencv_util.hpp"

using namespace cv;

// From: http://www.shervinemami.info/imageTransforms.html
// Rotate the image clockwise (or counter-clockwise if negative).
// Remember to free the returned image.
IplImage *rotateImage(const IplImage *src, float rotation_angle)
{
	// Create a map_matrix, where the left 2x2 matrix
	// is the transform and the right 2x1 is the dimensions.
	float m[6];
	CvMat M = cvMat(2, 3, CV_32F, m);
	int w = src->width;
	int h = src->height;
	float angleRadians = rotation_angle * ((float)CV_PI / 180.0f);
	m[0] = (float)( cos(angleRadians) );
	m[1] = (float)( sin(angleRadians) );
	m[3] = -m[1];
	m[4] = m[0];
	m[2] = w*0.5f;  
	m[5] = h*0.5f;

	// Make a spare image for the result
	CvSize sizeRotated;
	sizeRotated.width = cvRound(w);
	sizeRotated.height = cvRound(h);

	// Rotate
	IplImage *imageRotated = cvCreateImage( sizeRotated,
		src->depth, src->nChannels );

	// Transform the image
	cvGetQuadrangleSubPix( src, imageRotated, &M);

	return imageRotated;
}

// Crop or pad an image
IplImage *cropPadImage(const IplImage *src, int roi_width, int roi_height,
					   unsigned int rng_w, unsigned int rng_h, CvScalar fillval)
{
  int depth = src->depth;
  int nChannels = src->nChannels;
  int width = src->width;
  int height = src->height;

  // copy the source image
  IplImage *src_ = cvCreateImage(cvSize(width, height),
                                    depth, nChannels);
  cvCopy(src, src_, NULL);

  // deal with h direction
  int h_off = 0, w_off = 0;
  IplImage *img_crop_pad_h = cvCreateImage(cvSize(width, roi_height),
                                    depth, nChannels);
  cvSet(img_crop_pad_h, fillval);
  if (height > roi_height) {
	// crop in h direction
	h_off = rng_h % (height - roi_height);
	cvSetImageROI(src_, cvRect(0, h_off, width, roi_height));
  } else if (height < roi_height) {
    // pad in h direction
	h_off = rng_h % (roi_height - height);
	cvSetImageROI(img_crop_pad_h, cvRect(0, h_off, width, height));
  }
  cvCopy(src_, img_crop_pad_h, NULL);
  cvResetImageROI(img_crop_pad_h);
  cvReleaseImage(&src_);

  // deal with w direction
  IplImage *img_crop_pad_w = cvCreateImage(cvSize(roi_width, roi_height),
                                    depth, nChannels);
  cvSet(img_crop_pad_w, fillval);
  if (width > roi_width) {
    // crop in w direction
	w_off = rng_w % (width - roi_width);
	cvSetImageROI(img_crop_pad_h, cvRect(w_off, 0, roi_width, roi_height));
  } else if (width < roi_width) {
	// pad in w direction
	w_off = rng_w % (roi_width - width);
	cvSetImageROI(img_crop_pad_w, cvRect(w_off, 0, width, roi_height));
  }
  cvCopy(img_crop_pad_h, img_crop_pad_w, NULL);
  cvReleaseImage(&img_crop_pad_h);
  cvResetImageROI(img_crop_pad_w);

  return img_crop_pad_w;
}

// Shearing image in x and y direction
IplImage *shearImage(const IplImage *src, float shearing_ratio_x, float shearing_ratio_y,
					 int interpolation, CvScalar fillval)
{
  int depth = src->depth;
  int nChannels = src->nChannels;
  int width = src->width;
  int height = src->height;

  // map matrix
  float m[6];
  CvMat M = cvMat(2, 3, CV_32F, m);
  m[0] = 1;
  m[1] = shearing_ratio_y;
  m[2] = 0;
  m[3] = shearing_ratio_x;
  m[4] = 1;
  m[5] = 0;

  // create output image to be large enough to hold the transformed image
  int maxXOffset = abs(height * shearing_ratio_y);
  int maxYOffset = abs(width * shearing_ratio_x);
  int img_shear_width = width + maxXOffset;
  int img_shear_height = height + maxYOffset;  
  IplImage *img_shear = cvCreateImage(cvSize(img_shear_width, img_shear_height),
                                    depth, nChannels);

  // shear the image
  cvWarpAffine(src, img_shear, &M,
	  interpolation+CV_WARP_FILL_OUTLIERS, fillval);

  return img_shear;
}

// Get Shearing-->Resizing-->Rotation in-one-go transform matrix
void getShearResizeRotateTransform(int src_width, int src_height,
								   int dst_width, int dst_height,
								   float shearing_ratio_x, float shearing_ratio_y,
								   float rotation_angle,
								   CvMat *map_matrix)
{

  // scaling factor
  int maxXOffset = abs(src_height * shearing_ratio_y);
  int maxYOffset = abs(src_width * shearing_ratio_x);
  int img_shear_width = src_width + maxXOffset;
  int img_shear_height = src_height + maxYOffset;  
  float sx = ((float)dst_width) / img_shear_width;
  float sy = ((float)dst_height) / img_shear_height;

  // rotation angle
  float angleRadians = rotation_angle * ((float)CV_PI / 180.0f);
  angleRadians *= -1; // this is for checking correctness with previous seperately implementation
  float c = (float)( cos(angleRadians) );
  float s = (float)( sin(angleRadians) );
  // rotation center
  CvPoint2D32f center;
  center.x = (float)dst_width * 0.5f;
  center.y = (float)dst_height * 0.5f;

  // map matrix
  // A_rotation * A_resizing * A_shearing = 
  // Also see getRotationMatrix2D for reference
  CV_MAT_ELEM(*map_matrix, float, 0, 0) = sx * c + sy * shearing_ratio_x * s;
  CV_MAT_ELEM(*map_matrix, float, 0, 1) = sx * shearing_ratio_y * c + sy * s;
  CV_MAT_ELEM(*map_matrix, float, 0, 2) = (1 - c) * center.x - s * center.y; // translation
  CV_MAT_ELEM(*map_matrix, float, 1, 0) = - sx * s + sy * shearing_ratio_x * c;
  CV_MAT_ELEM(*map_matrix, float, 1, 1) = - sx * shearing_ratio_y * s + sy * c;
  CV_MAT_ELEM(*map_matrix, float, 1, 2) = s * center.x + (1 - c) * center.y; // translation
  CV_MAT_ELEM(*map_matrix, float, 2, 0) = 0;
  CV_MAT_ELEM(*map_matrix, float, 2, 1) = 0;
  CV_MAT_ELEM(*map_matrix, float, 2, 2) = 1;

}

// Shearing-->Resizing-->Rotation in one go
IplImage *shearResizeRotateImage(const IplImage *src,
								 int dst_width, int dst_height,
								 float shearing_ratio_x, float shearing_ratio_y,
								 float rotation_angle, 
								 int interpolation, CvScalar fillval)
{
  int depth = src->depth;
  int nChannels = src->nChannels;
  int width = src->width;
  int height = src->height;

  // scaling factor
  int maxXOffset = abs(height * shearing_ratio_y);
  int maxYOffset = abs(width * shearing_ratio_x);
  int img_shear_width = width + maxXOffset;
  int img_shear_height = height + maxYOffset;  
  float sx = ((float)dst_width) / img_shear_width;
  float sy = ((float)dst_height) / img_shear_height;

  // rotation angle
  float angleRadians = rotation_angle * ((float)CV_PI / 180.0f);
  angleRadians *= -1; // this is for checking correctness with previous seperately implementation
  float c = (float)( cos(angleRadians) );
  float s = (float)( sin(angleRadians) );
  // rotation center
  CvPoint2D32f center;
  center.x = (float)dst_width * 0.5f;
  center.y = (float)dst_height * 0.5f;

  // map matrix
  float m[6];
  CvMat M = cvMat(2, 3, CV_32F, m);
  // A_rotation * A_resizing * A_shearing = 
  // Also see getRotationMatrix2D for reference
  m[0] = sx * c + sy * shearing_ratio_x * s;
  m[1] = sx * shearing_ratio_y * c + sy * s;
  m[2] = (1 - c) * center.x - s * center.y; // translation
  m[3] = - sx * s + sy * shearing_ratio_x * c;
  m[4] = - sx * shearing_ratio_y * s + sy * c;
  m[5] = s * center.x + (1 - c) * center.y; // translation

  // Make a spare image for the result
  IplImage *dst = cvCreateImage( cvSize(dst_width, dst_height),
		            src->depth, src->nChannels );
  // shearing-->resizing-->rotation in one go!
  cvWarpAffine(src, dst, &M,
	  interpolation+CV_WARP_FILL_OUTLIERS, fillval);

  return dst;
}


void getPersepctiveTransform(int width, int height, 
							 float *perspective_ratio_x,
							 float *perspective_ratio_y,
							 CvMat *map_matrix)
{
	// four pairs of points
	CvPoint2D32f srcTri[4], dstTri[4];

    // src points
	srcTri[0].x = width * perspective_ratio_x[0];
	srcTri[0].y = height * perspective_ratio_y[0];
	srcTri[1].x = width * perspective_ratio_x[1];
	srcTri[1].y = height * perspective_ratio_y[1];
	srcTri[2].x = width * perspective_ratio_x[2];
	srcTri[2].y = height * perspective_ratio_y[2];
	srcTri[3].x = width * perspective_ratio_x[3];
	srcTri[3].y = height * perspective_ratio_y[3];

	// dst points
	dstTri[0].x = 0;
	dstTri[0].y = 0;
	dstTri[1].x = width;
	dstTri[1].y = 0;
	dstTri[2].x = 0;
	dstTri[2].y = height;
	dstTri[3].x = width;
	dstTri[3].y = height;

	// map matrix
	cvGetPerspectiveTransform( srcTri, dstTri, map_matrix );

}


IplImage *warpPerspective(const IplImage *src,
						  float *perspective_ratio_x,
						  float *perspective_ratio_y,
						  int interpolation, CvScalar fillval)
{
	float m[9];
	CvMat perspective_matrix = cvMat(3, 3, CV_32F, m);
	getPersepctiveTransform(src->width, src->height,
		perspective_ratio_x, perspective_ratio_y, &perspective_matrix);

	// Make a spare image for the result
    IplImage *dst = cvCreateImage( cvSize(src->width, src->height),
		            src->depth, src->nChannels );
	
	cvWarpPerspective( src, dst, &perspective_matrix,
		interpolation+CV_WARP_FILL_OUTLIERS, fillval);

	return dst;
}

// Shearing-->Resizing-->Rotation-->Perspective in-one-go
IplImage *warpPerspectiveOneGo(const IplImage *src,
							   int dst_width, int dst_height,
							   float shearing_ratio_x, float shearing_ratio_y,
							   float rotation_angle,
							   float *perspective_ratio_x,
							   float *perspective_ratio_y,
							   int interpolation, CvScalar fillval)
{
	// get Shearing-->Resizing-->Rotation in-one-go transform matrix
	float m1[9];
	CvMat shear_resize_rotate_matrix = cvMat(3, 3, CV_32F, m1);
	getShearResizeRotateTransform(src->width, src->height, dst_width, dst_height,
		shearing_ratio_x, shearing_ratio_y, rotation_angle, &shear_resize_rotate_matrix);

	// get perspective matrix
	float m2[9];
	CvMat perspective_matrix = cvMat(3, 3, CV_32F, m2);
	getPersepctiveTransform(dst_width, dst_height,
		perspective_ratio_x, perspective_ratio_y, &perspective_matrix);

	// compute the composite matrix
	Mat A = Mat(&shear_resize_rotate_matrix, true);
	Mat B = Mat(&perspective_matrix, true);
	Mat C = B * A; // mind the order
	CvMat map_matrix = (CvMat)C;

	// Make a spare image for the result
    IplImage *dst = cvCreateImage( cvSize(dst_width, dst_height),
		src->depth, src->nChannels );
	
	cvWarpPerspective( src, dst, &map_matrix,
		interpolation+CV_WARP_FILL_OUTLIERS, fillval);

	return dst;
}

