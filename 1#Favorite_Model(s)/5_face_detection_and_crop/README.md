# Face Detection and Crop (FDC)
**Implementation/documentation/review by Taihui Li, research work under supervision of Vahan M. Misakyan**

This repo provides the API for face detection and face crop in either images or videos. The underlying technique being used in the API is [Viola-Jones algorithm](https://wiki2.org/en/Viola%E2%80%93Jones_object_detection_framework).


## Table of Contents

1. [Environment Setting Up](#1-environment-setting-up)<br>
     1.1 [Required Dependencies](#11-required-dependencies)<br>
     1.2 [Installation Guide](#12-installation-guide)<br>
2. [Scripts/Directories Introduction](#2-scriptsdirectories-introduction)
3. [Usage](#3-usage)
4. [State of The Art](#4-state-of-the-art)
5. [Further Reading](#5-further-reading)
6. [Reference](#reference)



## 1 Environment Setting Up

### 1.1 Required Dependencies

   * [opencv-python 4.1.0.25](https://pypi.org/project/opencv-python/).

* [Python3.6](https://www.python.org/download/releases/3.0/).

  

### 1.2 Installation Guide
1. Create a virtual environment named ```FDC``` (the benefit of using virtual environment can be found [here](https://www.geeksforgeeks.org/python-virtual-environment/)):

   ```
   $ conda create -n FDC python=3.6
   ```

2. Activate your virtual environment (all the following steps will be done in this activated virtual environment):

   ```
   $ source activate FDC 
   ```

   OR you can use:

   ```
   $ conda activate FDC
   ```

3. Install opencv-python:

   ```
   $ pip install opencv-python
   ```



## 2 Scripts/Directories Introduction

This section introduces the scripts and directories in this implement code. The directory structure tree is shown below:
```
.
├── images_input                   /* The input images (format doesn't matter).
├── videos_input                   /* The input videos (format doesn't matter).
├── img_video_detector_output      /* The detector results for images or videos.
├── img_video_crop_output          /* The crop results for images or videos.
├── state_of_the_art_papers        /* The directory where the state-of-the-art papers are resided.
├── face_detection_crop.py         /* The core code to finish the task of face detecion and face crop for either images or videos.
├── Useit.py                       /* The usage API.

```


## 3 Usage
The API is very easy to use. 

1. Place your images under ```images_input``` and your videos under ```videos_input```.

2. Run command ```python Useit.py [--options]```. 

   ```
   $ python Useit.py [--options]
   ```

   When one image is processed, a result image will show. You can enter *anykey* to continue.

```[--options]``` is list below:

```
'--scale_factor', type=float, default=1.6, help='How much the image size is reduced at each image scale '

'--min_neighbors', type=int, default=6, help='How many neighbors each candidate rectangle should have to retain it'

'--min_size_w', type=int, default=1, help='Minimum possible object size (width).'

'--min_size_h', type=int, default=1, help='Minimum possible object size (height).'

'--save_path_d', type=str, default='img_video_detector_output/', help='The path to save detector results'

'--save_path_c', type=str, default='img_video_crop_output/', help='The path to save crop results'

'--showflag', type=bool, default=True, help='True or False'

'--videocropflag', type=bool, default=True, help='True or False'

```



## 4 State of The Art

1. [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/pdf/1905.00641v2.pdf).
2. [Accurate Face Detection for High Performance](https://arxiv.org/pdf/1905.01585v3.pdf).
3. [PyramidBox: A Context-assisted Single Shot Face Detector](https://arxiv.org/pdf/1803.07737v2.pdf).
4. [DSFD: Dual Shot Face Detector](https://arxiv.org/pdf/1810.10220v3.pdf).
5. [Selective Refinement Network for High Performance Face Detection](https://arxiv.org/pdf/1809.02693v1.pdf).
6. [Can We Still Avoid Automatic Face Detection?](https://arxiv.org/pdf/1602.04504v1.pdf).
7. [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/pdf/1604.02878.pdf).

## 5 Further Reading

1. [OpenCV-Python Cheat Sheet: From Importing Images to Face Detection](https://heartbeat.fritz.ai/opencv-python-cheat-sheet-from-importing-images-to-face-detection-52919da36433).

## Reference
1. [Face Detection: A Survey](https://www.sciencedirect.com/science/article/pii/S107731420190921X).
2. [Rapid Object Detection Using a Boosted Cascade of Simple Features](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C24&q=Rapid+object+detection+using+a+boosted+cascade+of+simple+features&btnG=).



