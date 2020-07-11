# -*- coding: utf-8 -*-
"""
Created on Dec 26 2018
@author: Dongfang Yang
@email: yang.3455@osu.edu

These modules are used for pedestrian detection in drone video processing

"""
import cv2
import sys
import os
import numpy as np


def generate_background(path_video, max_frame=20000, method='GSOC'):
    '''
    Method   obtained_bg

    MOG2:    color, good, vehicle shadow
    KNN:     color, ok, vehicle particle shadow

    MOG:       N.A.
    CNT:     gray, ok, with shadow
    LSBP:    color, average
    GSOC:    color, very good
    GMG:       N.A.
    '''

    if method == 'GSOC':
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
    elif method == 'MOG2':
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=False)  # default history=500
    else:
        # bg_subtractor = cv2.createBackgroundSubtractorKNN(history=800, detectShadows=False) # default history=500
        # bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
        # bg_subtractor = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, maxPixelStability=200)
        # bg_subtractor = cv2.bgsegm.createBackgroundSubtractorLSBP()
        # bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120)
        bg_subtractor = None
        print('Undefined background subtraction method!')

    # open video
    video = cv2.VideoCapture(path_video)
    width = int(video.get(3))
    height = int(video.get(4))
    fps = video.get(5)
    frame_total = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video" )
        sys.exit()

    frame_index = 0
    # load new frame
    frame_background = None
    while frame_index < max_frame:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        frame_index = frame_index + 1
        print('Generating background of '+path_video+', frames processed: '+str(frame_index)+'/'+str(frame_total)+' ...\r')

        # cropping frame
        frame_fgmask = bg_subtractor.apply(frame)
        frame_background = bg_subtractor.getBackgroundImage()

    return frame_background