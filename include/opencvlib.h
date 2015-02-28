#if !defined _HEADER_OPENCV_LIB_20140313_INCLUDED_
#define _HEADER_OPENCV_LIB_20140313_INCLUDED_

#include "opencv2/opencv.hpp"
using namespace cv;

#define OPENCV_VERSION   CVAUX_STR(CV_VERSION_EPOCH) CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR)
#define OPENCV_LIB_PREFIX(module) "../../3rdparty/lib/" "opencv_" #module OPENCV_VERSION

#ifdef _DEBUG
#define OPENCV_LIB_PATH(module) OPENCV_LIB_PREFIX(module) "d.lib"
#else
#define OPENCV_LIB_PATH(module) OPENCV_LIB_PREFIX(module) ".lib"
#endif

#ifdef _MSC_VER
#define LINK_OPENCV_LIB(module) __pragma(comment(lib, OPENCV_LIB_PATH(module)))
#else
#define LINK_OPENCV_LIB(module) /##/ module
#endif

LINK_OPENCV_LIB("core")
LINK_OPENCV_LIB("imgproc")
LINK_OPENCV_LIB("highgui")
LINK_OPENCV_LIB("legacy")
LINK_OPENCV_LIB("objdetect")

#endif //_HEADER_OPENCV_LIB_20140313_INCLUDED_