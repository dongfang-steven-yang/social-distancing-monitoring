import torch
import torchvision
import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from plotting import plot_frame, plot_frame_one_row, get_roi_pts
from utils import ROIs, find_violation

from utils import COCO_INSTANCE_CATEGORY_NAMES as LABELS
import cv2
np.set_printoptions(precision=4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

detector = 'faster_rcnn'


def main(dataset, data_time, detector):

    path_result = os.path.join('results', data_time + '_' + detector, dataset)
    os.makedirs(path_result, exist_ok=True)

    # initialize detector
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device=device)
    model.eval()

    # load background
    img_bkgd_bev = cv2.imread('calibration/' + dataset + '_background_calibrated.png')
    # load transformation matrix
    transform_cam2world = np.loadtxt('calibration/' + dataset + '_matrix_cam2world.txt')

    # open video of dataset
    if dataset == 'oxford_town':
        cap = cv2.VideoCapture(os.path.join('datasets', 'TownCentreXVID.avi'))
        frame_skip = 10  # oxford town dataset has fps of 25
        thr_score = 0.9
    elif dataset == 'mall':
        cap = cv2.VideoCapture(os.path.join('datasets', 'mall.mp4'))
        frame_skip = 1
        thr_score = 0.9
    elif dataset == 'grand_central':
        cap = cv2.VideoCapture(os.path.join('datasets', 'grandcentral.avi'))
        frame_skip = 25  # grand central dataset has fps of 25
        thr_score = 0.5
    else:
        raise Exception('Invalid Dataset')

    # f = open(os.path.join(path_result, 'statistics.txt'), 'w')
    statistic_data = []
    i_frame = 0
    # while cap.isOpened() and i_frame < 5000:
    while cap.isOpened():
        ret, img = cap.read()
        # print('Frame %d - ' % i_frame)
        if ret is False:
            break

        # if i_frame > 50:
        #     break

        # skip frames to achieve 1hz detection
        # if not i_frame % frame_skip == 0:  # conduct detection per second
        #     i_frame += 1
        #     continue

        if i_frame / frame_skip < 20:
            vis = True
        else:
            vis = False

        # counting process time
        t0 = time.time()

        # convert image from OpenCV format to PyTorch tensor format
        img_t = np.moveaxis(img, -1, 0) / 255
        img_t = torch.tensor(img_t, device=device).float()

        # pedestrian detection
        predictions = model([img_t])
        boxes = predictions[0]['boxes'].cpu().data.numpy()
        classIDs = predictions[0]['labels'].cpu().data.numpy()
        scores = predictions[0]['scores'].cpu().data.numpy()

        # get positions and plot on raw image
        pts_world = []
        for i in range(len(boxes)):
            if classIDs[i] == 1 and scores[i] > thr_score:
                # extract the bounding box coordinates
                (x1, y1) = (boxes[i][0], boxes[i][1])
                (x2, y2) = (boxes[i][2], boxes[i][3])

                if vis:
                    # draw a bounding box rectangle and label on the image
                    cv2.rectangle(img, (x1, y1), (x2, y2), [0, 0, 255], 2)
                    text = "{}: {:.2f}".format(LABELS[classIDs[i]], scores[i])
                    cv2.putText(img, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)

                # find the bottom center position and convert it to world coordinate
                p_c = np.array([[(x1 + x2)/2], [y2], [1]])
                p_w = transform_cam2world @ p_c
                p_w = p_w / p_w[2]
                pts_world.append([p_w[0][0], p_w[1][0]])

        t1 = time.time()

        pts_world = np.array(pts_world)
        if dataset == 'oxford_town':
            pts_world[:, [0, 1]] = pts_world[:, [1, 0]]
            pass

        elif dataset == 'mall':
            # pts_world[:, [0, 1]] = pts_world[:, [1, 0]]
            pass
        elif dataset == 'grand_central':
            # pts_world[:, [0, 1]] = pts_world[:, [1, 0]]
            pass

        statistic_data.append((i_frame, t1 - t0, pts_world))

        # visualize
        if vis:
            violation_pairs = find_violation(pts_world)
            pts_roi_world, pts_roi_cam = get_roi_pts(dataset=dataset, roi_raw=ROIs[dataset], matrix_c2w=transform_cam2world)

            fig = plot_frame_one_row(
                dataset=dataset,
                img_raw=img,
                pts_roi_cam=pts_roi_cam,
                pts_roi_world=pts_roi_world,
                pts_w=pts_world,
                pairs=violation_pairs
            )

            # fig = plot_frame(
            #     dataset=dataset,
            #     img_raw=img,
            #     img_bev_bkgd_10x=img_bkgd_bev,
            #     pts_roi_cam=pts_roi_cam,
            #     pts_roi_world=pts_roi_world,
            #     pts_w=pts_world,
            #     pairs=violation_pairs
            # )

            fig.savefig(os.path.join(path_result, 'frame%04d.png' % i_frame))
            plt.close(fig)

        # update loop info
        print('Frame %d - Inference Time: %.2f' % (i_frame, t1 - t0))
        print('=======================')
        i_frame += 1

    # save statistics
    # f.close()
    pickle.dump(statistic_data, open(os.path.join(path_result, 'statistic_data.p'), 'wb'))


if __name__ == '__main__':
    data_time = 'test'
    # data_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets = ['oxford_town', 'grand_central', 'mall']
    # datasets = ['oxford_town']

    for dataset in datasets:
        print('=========== %s ===========' % dataset)
        main(dataset=dataset, data_time=data_time, detector=detector)