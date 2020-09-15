# social-distancing-monitoring
A Vision-based Social Distancing and Critical Density Detection System for COVID-19

<img src="images/osu-logo.jpg" width="300">

Developed by Dongfang Yang and Ekim Yurtsever at [Control and Intelligent Transportation Research (CITR) Lab](http://citr.osu.edu/), The Ohio State University.

Paper: [arXiv preprint](https://arxiv.org/abs/2007.03578)

**Updata Log:**
- **2020.09.15**: We have added the instructions for running the pedestrian detection and the data analysis. Please see the section of 'Getting Started'.
- **2020.07.10**: We are still in the process of finalizing the repository. The complete version will be released soon!

## System Overview

Our system is real-time and does not record data. An audio-visual cue is emitted each time an individual breach of social distancing is detected. We also make a novel contribution by defining a critical social density value for measuring overcrowding. Entrance into the region-of-interest can be modulated with this value.

![scenario](images/overview.png)

## Social Distancing Monitoring

Illustration of pedestrian detection and social distancing monitoring.

![scenario](images/grand_central.gif)

![scenario](images/oxford.gif)

![scenario](images/mall.gif)

## Camera calibration

NYC Grand Central terminal: We found the floor plan of the building and calibrated the camera by picking landmarks. We provide the transformation matrix in
`calibration/grand_central_matrix_cam2world.txt`

Oxford town center: The original dataset provides the transformation matrix. We added it here also `calibration/oxford_town_matrix_cam2world.txt`

Mall: We could not found the transformation matrix or the floor plan of this dataset. Instead, we first estimated the size of a reference object in the image by comparing it with the width of detected  pedestrians  and  then  utilized  the  key  points  of  the reference object to calculate the perspective transformation. We provide this transformation matrix in `calibration/mall_matrix_cam2world.txt`

## Critical Density

Keeping the social density under the critical value will keep the probability of social distancing violation occurrence near zero with the linear regression assumption.

![scenario](images/critical_density.png)

## Getting Started

### 1. Environment Config
The program was developed based on `python 3.7` with `pytorch 1.5`.
We highly recommend to use `conda`. After you have created a new conda environment, use the following command to install pytorch:
```shell
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```

You may also need to install the packages specified in `requirements.txt` file if you don't have them.

### 2. Download Dataset

We provide an alternative link for you to download all the datasets: https://drive.google.com/file/d/1G6nZS-EZLrNBC68CRDf-yo2uj3k32j35/view?usp=sharing

Then copy all the files into folder `datasets` in the repository.

### 3. Run Pedestrian Detection

Execute `detect.py` to obtain the detection result. The result will be saved at folder `results` in the repository. Result will be saved as pickle `.p` file for each dataset.

### 4. Run Analysis

Execute `analyze.py` to obtain the analysis result. It will be saved in the same folder `results`.


## TODO Lists

- [x] Social distancing monitoring pipeline
- [x] Evaluation on different pedestrian crowd datasets
- [x] Detector: Faster R-CNN
- [x] Detector: Yolo v4
- [ ] Detector: EfficientDet
- [x] Critical density analysis
- [ ] Embedded system integration
- [ ] Camera calibration UI


## Contact

For further info please contact:
- Dongfang Yang: yang.3455@osu.edu
- Ekim Yurtsever: yurtsever.2@osu.edu
