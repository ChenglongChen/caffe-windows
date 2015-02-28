#include <string>

#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

using namespace cv;

// Rotate the image clockwise (or counter-clockwise if negative).
// Remember to free the returned image.
IplImage *rotateImage(const IplImage *src, float rotation_angle);

// Crop or pad the image
IplImage *cropPadImage(const IplImage *src, int roi_width, int roi_height,
					   unsigned int rng_w, unsigned int rng_h, CvScalar fillval);

// Shear the image
IplImage *shearImage(const IplImage *src, float offset_x, float offset_y,
					 int interpolation, CvScalar fillval);


// Get Shearing-->Resizing-->Rotation in-one-go transform matrix
void getShearResizeRotateTransform(int src_width, int src_height,
								   int dst_width, int dst_height,
								   float shearing_ratio_x, float shearing_ratio_y,
								   float rotation_angle,
								   CvMat *map_matrix);


// Shearing-->Resizing-->Rotation in one go
IplImage *shearResizeRotateImage(const IplImage *src,
								 int dst_width, int dst_height,
								 float shearing_ratio_x, float shearing_ratio_y,
								 float rotation_angle, 
								 int interpolation, CvScalar fillval);



// get perspective transform matrix
void getPersepctiveTransform(int width, int height, 
							 float *perspective_ratio_x,
							 float *perspective_ratio_y,
							 CvMat *map_matrix);

// warp perspective
IplImage *warpPerspective(const IplImage *src,
						  float *perspective_ratio_x,
						  float *perspective_ratio_y,
						  int interpolation, CvScalar fillval);

// Shearing-->Resizing-->Rotation-->Perspective in-one-go
IplImage *warpPerspectiveOneGo(const IplImage *src,
							   int dst_width, int dst_height,
							   float shearing_ratio_x, float shearing_ratio_y,
							   float rotation_angle,
							   float *perspective_ratio_x,
							   float *perspective_ratio_y,
							   int interpolation, CvScalar fillval);

//*/
/*
// See types_c.h
// Sub-pixel interpolation methods
enum
{
    CV_INTER_NN        =0,
    CV_INTER_LINEAR    =1,
    CV_INTER_CUBIC     =2,
    CV_INTER_AREA      =3,
    CV_INTER_LANCZOS4  =4
};
*/

/*
// See types_c.h
// Image smooth methods
enum
{
    CV_BLUR_NO_SCALE =0,
    CV_BLUR  =1,
    CV_GAUSSIAN  =2,
    CV_MEDIAN =3,
    CV_BILATERAL =4
};
*/
