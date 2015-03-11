#include <string>

#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/util/opencv_util.hpp"

using namespace cv;
namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::TransformSingle(const int batch_item_id,
                                       IplImage *img,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  int channels = img->nChannels;
  int width = img->width;
  int height = img->height;
  unsigned char* data = (unsigned char *)img->imageData;
  int step = img->widthStep / sizeof(char);
  // crop 4 courners + center
  int w[5], h[5];
  FillInOffsets(w, h, width, height, crop_size);
  int h_off, w_off;
  // We only do random crop when we do training.
  if (phase_ == Caffe::TRAIN) {
    int r = Rand() % 5;
    h_off = h[r];
    w_off = w[r];
  } else {
    h_off = h[4];
    w_off = w[4];
  }


  ////// -------------------!! for debug !! -------------------
  // IplImage *dest = cvCreateImage(cvSize(crop_size * 2, crop_size * 2),
                                    // img->depth, img->nChannels);
  // cvResize(img, dest);
  // cvNamedWindow("Sample1");
  // cvNamedWindow("Sample2");
  // if (phase_ == Caffe::TRAIN)
  //   cvShowImage("Sample", dest);
  // else
  //   cvShowImage("Sample", img);
  // cvWaitKey(0);
  // cvReleaseImage(&img);
  // cvReleaseImage(&dest);
  // if (phase_ == Caffe::TRAIN) {
  //   cvSetImageROI(img, cvRect(w_off, h_off, crop_size, crop_size));
  //   // cvCopy(img, dest, NULL);
  //   cvResize(img, dest);
  //   cvResetImageROI(img);
  //   cvShowImage("Sample1", img);
  //   cvShowImage("Sample2", dest);
  //   cvWaitKey(0);
  // }
  // cvReleaseImage(&dest);
  ////// -------------------------------------------------------
  if (mirror && Rand() % 2) {
    // Copy mirrored version
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < crop_size; h++) {
        for (int w = 0; w < crop_size; w++) {
          int top_index = ((batch_item_id * channels + c) * crop_size + h)
                          * crop_size + (crop_size - 1 - w);
          int data_index = (h + h_off) * step + (w + w_off) * channels + c;
          int mean_index = (c * crop_size + h) * crop_size + w;
          Dtype datum_element = static_cast<Dtype>(data[data_index]);
          transformed_data[top_index] = (datum_element - mean[mean_index]) * scale;
        }
      }
    }
  } else {
    // Normal copy
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < crop_size; h++) {
        for (int w = 0; w < crop_size; w++) {
          int top_index = ((batch_item_id * channels + c) * crop_size + h)
                          * crop_size + w;
          int data_index = (h + h_off) * step + (w + w_off) * channels + c;
          int mean_index = (c * crop_size + h) * crop_size + w;
          Dtype datum_element = static_cast<Dtype>(data[data_index]);
          transformed_data[top_index] = (datum_element - mean[mean_index]) * scale;
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformMultiple(const int batch_item_id,
                                       IplImage *img,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();
  const bool debug_display = param_.debug_display();
  const bool contrast_adjustment = param_.contrast_adjustment();
  const bool smooth_filtering = param_.smooth_filtering();
  const bool jpeg_compression = param_.jpeg_compression();

  // param for scaling
  const float min_scaling_factor = param_.min_scaling_factor();
  const float max_scaling_factor = param_.max_scaling_factor();
  // param for rotation
  const float angle_interval = param_.angle_interval();
  // param for shearing
  const float max_shearing_ratio = param_.max_shearing_ratio();
  // param for perspective warpping
  const float max_perspective_ratio = param_.max_perspective_ratio();
  // border fill value
  const CvScalar warp_fillval = cvScalarAll(param_.warp_fillval());

  // figure out dimension
  int channels = img->nChannels;
  int width = img->width;
  int height = img->height;

  if (debug_display && phase_ == Caffe::TRAIN)
	  cvShowImage("Source", img);

  // Flipping and Reflection -----------------------------------------------------------------
  int flipping_mode = (Rand() % 4) - 1; // -1, 0, 1, 2
  bool apply_flipping = flipping_mode != 2;
  if (apply_flipping) {
	  cvFlip(img, NULL, flipping_mode);
	  if (debug_display && phase_ == Caffe::TRAIN)
		  cvShowImage("Flipping and Reflection", img);
  }

  // Smooth Filtering -------------------------------------------------------------
  int smooth_type = 0, smooth_param1 = 3;
  int apply_smooth = Rand() % 2;
  if ( smooth_filtering && apply_smooth ) {
	smooth_type = Rand() % 4; // see opencv_util.hpp
	smooth_param1 = 3 + 2*(Rand() % 1);
	cvSmooth(img, img, smooth_type, smooth_param1);
	if (debug_display && phase_ == Caffe::TRAIN)
      cvShowImage("Smooth Filtering", img);
  }

  // Contrast and Brightness Adjuestment ----------------------------------------
  float alpha = 1, beta = 0;
  int apply_contrast = Rand() % 2;
  if ( contrast_adjustment && apply_contrast ) {
    float min_alpha = 0.8, max_alpha = 1.2;
	alpha = Uniform(min_alpha, max_alpha);
    beta = (float)(Rand() % 6);
	// flip sign
	if ( Rand() % 2 ) beta = - beta;
    cvConvertScale(img, img, alpha, beta);
	if (debug_display && phase_ == Caffe::TRAIN)
      cvShowImage("Contrast Adjustment", img);
  }

  // JPEG Compression -------------------------------------------------------------
  // DO NOT use the following code as there is some memory leak which I cann't figure out
  int QF = 100;
  int apply_JPEG = Rand() % 2;
  if ( jpeg_compression && apply_JPEG ) {
	// JPEG quality factor
	QF = 95 + 1 * (Rand() % 6);
	int compression_params[2] = {CV_IMWRITE_JPEG_QUALITY, QF};
    CvMat *img_jpeg = cvEncodeImage(".jpg", img, compression_params);
	img = cvDecodeImage(img_jpeg);
	cvReleaseMat(&img_jpeg);
	if (debug_display && phase_ == Caffe::TRAIN)
      cvShowImage("JPEG Compression", img);
  }

  // Histogram Equalization ------------------------------------------------
  //if ( Rand() % 2 )
  //	cvEqualizeHist(img, img);

  // Cropping and Padding -----------------------------------------------------------------  
  /* Since in the end, we will resize the image to a fixed size (i.e. crop_size), so scaling
   * the image will not make any difference. Therefore, we use cropping and padding to
   * simulate scaling effect.
   * For scaling factor > 1, we random crop the original image to simulate scaling up
   * For scaling factor < 1, we random pad the original image to simulate scaling down
  */
  // scaling factor for height and width respectively
  float sf_w = Uniform(min_scaling_factor, max_scaling_factor);
  float sf_h = Uniform(min_scaling_factor, max_scaling_factor);
  // ROI height and width
  int roi_width = (int)(width * (1. / sf_w));
  int roi_height = (int)(height * (1. / sf_h));
  // random number for w_off and h_oof in cropPadImage function
  unsigned int rng_w = Rand(), rng_h = Rand();
  IplImage *img_crop_pad = cropPadImage(img, roi_width, roi_height,
	                                    rng_w, rng_h, warp_fillval);
  if (debug_display && phase_ == Caffe::TRAIN)
      cvShowImage("Cropping and Padding", img_crop_pad);

  // param config for shearing
  float shearing_ratio_x = Uniform(-max_shearing_ratio, max_shearing_ratio);
  float shearing_ratio_y = Uniform(-max_shearing_ratio, max_shearing_ratio);

  // param config for rotation
  float angle_raw = Uniform(0, 360);
  float angle_quant = angle_interval * ceil(angle_raw / angle_interval + 0.5);

  // param for perspective warpping
  // for x of four points
  float perspective_ratio_x[4];
  perspective_ratio_x[0] = 0 + Uniform(-max_perspective_ratio, max_perspective_ratio); // top-left
  perspective_ratio_x[1] = 1 + Uniform(-max_perspective_ratio, max_perspective_ratio); // top-right
  perspective_ratio_x[2] = 0 + Uniform(-max_perspective_ratio, max_perspective_ratio); // bottom-left
  perspective_ratio_x[3] = 1 + Uniform(-max_perspective_ratio, max_perspective_ratio); // bottom-right
  // for y of four points
  float perspective_ratio_y[4];
  perspective_ratio_y[0] = 0 + Uniform(-max_perspective_ratio, max_perspective_ratio); // top-left
  perspective_ratio_y[1] = 0 + Uniform(-max_perspective_ratio, max_perspective_ratio); // top-right
  perspective_ratio_y[2] = 1 + Uniform(-max_perspective_ratio, max_perspective_ratio); // bottom-left 
  perspective_ratio_y[3] = 1 + Uniform(-max_perspective_ratio, max_perspective_ratio); // bottom-right

  // random interpolation kernel
  int interpolation = Rand() % 5; // see opencv_util.hpp

  // perform perspective warpping in one go
  IplImage *dest = warpPerspectiveOneGo(img_crop_pad, crop_size, crop_size,
	  shearing_ratio_x, shearing_ratio_y, angle_quant, perspective_ratio_x, perspective_ratio_y,
	  interpolation, warp_fillval);
  if (debug_display && phase_ == Caffe::TRAIN)
      cvShowImage("Warp Perspective in one go", dest);
  

  //--------------------!! for debug only !!-------------------
  if (debug_display && phase_ == Caffe::TRAIN) {
	LOG(INFO) << "----------------------------------------";
	LOG(INFO) << "src width: " << width << ", src height: " << height;
	LOG(INFO) << "dest width: " << crop_size << ", dest height: " << crop_size;
	if (apply_flipping) {
		LOG(INFO) << "* parameter for flipping: ";
		LOG(INFO) << "  flipping_mode: " << flipping_mode;
	}
	if ( smooth_filtering && apply_smooth ) {
      LOG(INFO) << "* parameter for smooth filtering: ";
	  LOG(INFO) << "  smooth type: " << smooth_type << ", smooth param1: " << smooth_param1;
	}
	if ( contrast_adjustment && apply_contrast ) {
	  LOG(INFO) << "* parameter for contrast adjustment: ";
	  LOG(INFO) << "  alpha: " << alpha << ", beta: " << beta;
	}
	if ( jpeg_compression && apply_JPEG ) {
	  LOG(INFO) << "* parameter for JPEG compression: ";
	  LOG(INFO) << "  QF: " << QF;
	}
	LOG(INFO) << "* parameter for cropping and padding: ";
	LOG(INFO) << "  sf_w: " << sf_w << ", sf_h: " << sf_h;
	LOG(INFO) << "  roi_width: " << roi_width << ", roi_height: " << roi_height;
	LOG(INFO) << "* parameter for shearing: ";
	LOG(INFO) << "  max_shearing_ratio: " << max_shearing_ratio;
	LOG(INFO) << "  shearing_ratio_x: " << shearing_ratio_x << ", shearing_ratio_y: " << shearing_ratio_y;
	LOG(INFO) << "* parameter for rotation: ";
	LOG(INFO) << "  angle_interval: " << angle_interval;
	LOG(INFO) << "  angle: " << angle_quant;
	LOG(INFO) << "* parameter for perspective transform:";
	LOG(INFO) << "  max_perspective_ratio: " << max_perspective_ratio;
	LOG(INFO) << "  top-left:     [" << perspective_ratio_x[0] << "," << perspective_ratio_y[0] << "]";
	LOG(INFO) << "  top-right:    [" << perspective_ratio_x[1] << "," << perspective_ratio_y[1] << "]";
	LOG(INFO) << "  bottom-left:  [" << perspective_ratio_x[2] << "," << perspective_ratio_y[2] << "]";
	LOG(INFO) << "  bottom-right: [" << perspective_ratio_x[3] << "," << perspective_ratio_y[3] << "]";
	LOG(INFO) << "* parameter for in-one-go warpping : ";
	LOG(INFO) << "  interpolation: " << interpolation;
    cvWaitKey(0);
  }

  unsigned char* data = (unsigned char *)dest->imageData;
  int step = dest->widthStep / sizeof(char);
  // leave flipping to opencv

    // Normal copy
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < crop_size; h++) {
        for (int w = 0; w < crop_size; w++) {
          int top_index = ((batch_item_id * channels + c) * crop_size + h)
                          * crop_size + w;
          int data_index = h * step + w * channels + c;
          int mean_index = (c * crop_size + h) * crop_size + w;
          Dtype datum_element = static_cast<Dtype>(data[data_index]);
          transformed_data[top_index] = (datum_element - mean[mean_index]) * scale;
        }
      }
    }

  cvReleaseImage(&img_crop_pad);
  cvReleaseImage(&dest);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       IplImage *img,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  if (!param_.multiscale())
    TransformSingle(batch_item_id, img, mean, transformed_data);
  else
    TransformMultiple(batch_item_id, img, mean, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  // Rand() is always on for multiscale
  const bool needs_rand = ((phase_ == Caffe::TRAIN) && 
	  (param_.mirror() || param_.crop_size())) ||
	  param_.multiscale();
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

template <typename Dtype>
float DataTransformer<Dtype>::Uniform(const float min, const float max) {
  CHECK(rng_);
  Dtype d[1];
  caffe_rng_uniform<Dtype>(1, Dtype(min), Dtype(max), d);
  return (float)d[0];
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
