# Caffe Windows
Based on [@terrychenism](https://github.com/terrychenism)'s [caffe-windows-cudnn](https://github.com/terrychenism/caffe-windows-cudnn) with the following major changes.

Linux: Have a look at [@Senecaur](https://github.com/senecaur/caffe-rta)'s version [here](https://github.com/senecaur/caffe-rta). 

Note: This implementation here is for my project in [Kaggle National Data Science Bowl](https://www.kaggle.com/c/datasciencebowl). So, some choices in the code maybe specifc to the problem, and don't represent the general one, e.g., stochastic prediction as mentioned below.

## COMPACT_DATA layer to hold varying size images
This is modified from the [Princeton's GoogLeNet patch](http://vision.princeton.edu/pvt/GoogLeNet/code/).

### Usage
To use this layer, you have to convert the image to compact version of leveldb after building `/bin/convert_imageset_compact.exe` (the usage is the same with `/bin/convert_imageset.exe`).

Since the image can be of varying sizes, it might be problem when computing the mean image for this layer. I use the following method for this issue and it works ok.
- decide the final image size (`crop_size` as mentioned below) you want to input to the net, say `32x32`
- use `/bin/convert_imageset.exe` to pack the image in normal leveldb format with resizing option on, e.g.,

  ```
  ./bin/convert_imageset.exe \
      --resize_height=32 \
      --resize_width=32 \
      --gray \
      path-to-image-folder \
      path-to-image-list \
      path-to-leveldb-32x32
  ```
- use `/bin/compute_image_mean.exe` to compute the mean image, e.g.

  ```
  ./bin/compute_image_mean.exe path-to-leveldb-32x32 path-to-image-mean-32x32
  ```

### Note
In this code, I turn off the `iscolor` flag in the function call `cvDecodeImage` in [this line](https://github.com/ChenglongChen/caffe-windows/blob/master/src/caffe/layers/compact_data_layer.cpp#L187) and [this line](https://github.com/ChenglongChen/caffe-windows/blob/master/src/caffe/layers/compact_data_layer.cpp#L252). As a result, this layer will convert every image to grayscale. If you want color one, you can set `iscolor` to `1`.

## Realtime data augmentation
Realtime data augmentation is implemented within the `COMPACT_DATA` layer. It offers:
- Geometric transform: random flipping, cropping, resizing, rotation, shearing, perspective warpping
- Smooth filtering
- JPEG compression
- Contrast & brightness adjustment
- new can be added via OpenCV utils

### Usage
To use it, you can specify
```
## Training set
layers {
  name: "Image"
  type: COMPACT_DATA
  top: "data"
  top: "label"
  data_param {
    source: "path-to-training-compact-leveldb"
    batch_size: 100
  }
  transform_param {
    mean_file: "path-to-image-mean"
    mirror: true
    crop_size: 32
    multiscale: true
    debug_display: false  
    smooth_filtering: false
    jpeg_compression: false
    contrast_adjustment: false
    min_scaling_factor: 0.8
    max_scaling_factor: 1.2
    angle_interval: 45
    max_shearing_ratio: 0.1
    max_perspective_ratio: 0.1
    warp_fillval: 255
  }
  include: { phase: TRAIN }
}
## Validation set
layers {
  name: "Image"
  type: COMPACT_DATA
  top: "data"
  top: "label"
  data_param {
    source: "path-to-validation-compact-leveldb"
    batch_size: 100
  }
  transform_param {
    mean_file: "path-to-image-mean"
    mirror: true
    crop_size: 32
    multiscale: true
    debug_display: false	
    smooth_filtering: false
    jpeg_compression: false
    contrast_adjustment: false
    min_scaling_factor: 0.8
    max_scaling_factor: 1.2
    angle_interval: 45
    max_shearing_ratio: 0.1
    max_perspective_ratio: 0.1
    warp_fillval: 255
  }
  include: { phase: TEST }
}
```
### Parameter
Transformations parameter accepts parameters:
- `mirror`: horizontal, vertical flipping or both simultaneously
- `crop_size`: the final size of the image input to the net (after geometric tranformations, the image will be "resized" to this size); *This param has somewhat different meaning than in Caffe, but they both refer to the final size input to the net. In this code, cropping is carried out to simulate resizing; please see the explanation for the params min_scaling_factor \& max_scaling_factor below.*
- `multiscale`: to enable realtime data augmentation (param kept from the [Princeton's GoogLeNet patch](http://vision.princeton.edu/pvt/GoogLeNet/code/))
- `debug_display`: display the distorted image and some info for debugging purpose
- `smooth_filtering`: apply soomth filtering with varying filters (the choice of filters is currently hard coded but you can expose it)
- `jpeg_compression`: apply JPEG compression with varying QFs (the choice of QFs is currently hard coded but you can expose it)
- `contrast_adjustment`: apply contrast & brightness adjustment (the choice of the param, ie., `alpha` and `beta` is currently hard coded but you can expose it)
- `min_scaling_factor` and `max_scaling_factor`: perform random resizing with scaling factor uniformly sampled from `[min_scaling_factor, max_scaling_factor]`. Since in the end, we will resize the image to a fixed size (i.e. `crop_size`), so scaling the image will not make any difference. Therefore, we use cropping and padding to simulate scaling effect.
 - For scaling factor > 1, we random crop the original image to simulate scaling up
 - For scaling factor < 1, we random pad the original image to simulate scaling down
- `angle_interval`: perform random rotation with angle uniformly sampled from `[0, 360]` with step `angle_interval`
- `max_shearing_ratio`: perform random shearing with ratio uniformly sampled from `[-max_shearing_ratio, max_shearing_ratio]`
- `max_perspective_ratio`: perform random perspective warpping with ratio uniformly sampled from `[-max_perspective_ratio, max_perspective_ratio]`
- `warp_fillval`: value to fill the border pixels

Here is a concrete example about the geometric transformation. In the above prototxt config, let's say the net encounter an image with original size `48x60`, and the scaling factor for *h*(eight) and *w*(idth) direction is randomly sampled as `0.8` and `1.2`, which corresponds to a ROI of size `60x50` (*h*: `48/0.8=60`, *w*: `60/1.2=50`). In this case, for *h* direction, we will randomly `pad` additional `12` pixels in both side (these pixels will be set to `warp_fillval`); and for *w* direction, will randomly `crop` out extra `10` pixels on both side. With the resulted `60x50` ROI, we will perform random rotation/shearing/perspective warpping in combination using the function `warpPerspectiveOneGo` in `/src/caffe/util/opencv_util.cpp`. The output will then be a transformed image of size `32x32`. This is the image we feed to the net.
 
For a better understanding of the transformation augmentation and the above params, please see `/src/caffe/data_transformer.cpp` (the transformation is implemented here) and `/src/caffe/proto/caffe.proto`.

For transformation augmentation for image classification, I would like to recommend this paper: [Transformation Pursuit for Image Classification](https://hal.inria.fr/hal-00979464/document). The authors have a [project page](http://lear.inrialpes.fr/people/paulin/projects/ITP/) for it.

### Note
In this implemetnation, realtime augementation is always on in both `TRAIN` and `TEST` phase (even the `mirror` operation which is disabled in Caffe version). This suits the need for ensemble: you can run the trained model with the same input image a few times and average those predictions (they won't be the same due to random distortions) to get the final one. 

If you want deterministic prediction, you can hack the code or using something like:
```
## Validation set
layers {
  name: "Image"
  type: COMPACT_DATA
  top: "data"
  top: "label"
  data_param {
    source: "path-to-validation-compact-leveldb"
    batch_size: 100
  }
  transform_param {
    mean_file: "path-to-image-mean"
    mirror: false
    crop_size: 32
    multiscale: true
    debug_display: false  
    smooth_filtering: false
    jpeg_compression: false
    contrast_adjustment: false
    min_scaling_factor: 1
    max_scaling_factor: 1
    angle_interval: 360
    max_shearing_ratio: 0
    max_perspective_ratio: 0
    warp_fillval: 255
  }
  include: { phase: TEST }
}
```
Note the random mirroring is still on ;)

## Prediction module to get probability
It is within the same `/bin/caffe.exe` interface and usage is as follow:
```
# make prediction
./bin/caffe.exe predict \
  --model=path-to-model-prototxt \
  --weights=path-to-trained-model \
  --outfile=path-to-output-prediction \
  --label_number=number-of-label \
  --iterations=iteration-to-run \
  --score_index=which-score-to-output \
  --gpu=gpu-id \
  --random_seed=random-seed \
  --phase=TRAIN-or-TEST
```

## Batch Normalization layer
Batch Normalization is from [here](https://github.com/ChenglongChen/batch_normalization).

This implementation has be adopted in [this PR to Caffe](https://github.com/BVLC/caffe/pull/1965) (with improvements such as per mini-batch shuffling).

## PReLU layer and MSRA filler
PReLU is adopted from [this PR to Caffe](https://github.com/BVLC/caffe/pull/1940).

## AdaDelta solver
AdaDelta is based on [this PR to Caffe](https://github.com/BVLC/caffe/pull/1122) with a modification to allow learning rate policy as usual.

## Accumulated gradient method for SGD & Nesterov solver
Adopted from [Princeton's GoogLeNet patch](http://vision.princeton.edu/pvt/GoogLeNet/code/).