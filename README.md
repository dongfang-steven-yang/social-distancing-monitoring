# social-distancing-monitoring
CV-based social distancing monitoring and analyzing.

## Getting Started

(1) Download the video datasets:
   - Oxford Town Center Dataset: [Download](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi)
   - Mall Dataset
   - Grand Central Train Station Dataset:

(2) Put the dataset videos into `datasets` folder.

(3) Run `main.py` script.

## TODO Lists

- [x] Calibration of Oxford Town center dataset
- [x] Implement social distance monitoring
- [ ] Integrate a faster detector (e.g. SSD or Yolo) to replace faster RCNN.
- [ ] Real-time implementation: make the program run in real-time.
- [ ] Calibration of two additional datasets (train station, mall)
- [ ] Integrate two additional datasets
- [ ] Interface for any camera setup
