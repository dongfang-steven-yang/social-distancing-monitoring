import torch
import torchvision
import os
import numpy as np
import time
import matplotlib.pyplot as plt

from utils import COCO_INSTANCE_CATEGORY_NAMES as LABELS
import cv2
np.set_printoptions(precision=4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


def find_violation(ps, dist=2.0):
    n = len(ps)  # number of pedestrians
    pairs = []
    for i in np.arange(0, n, 1):
        for j in np.arange(i+1, n, 1):
            if np.linalg.norm(ps[i] - ps[j]) < dist:
                pairs.append((i, j))
    return pairs


def main():
    os.makedirs('results/', exist_ok=True)

    # initialize detector
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device=device)
    model.eval()

    # load background
    img_bkgd_bev = cv2.imread('calibration/oxford_town_background_calibrated.png')
    # load transformation matrix
    M_c2w = np.loadtxt('calibration/oxford_town_matrix_cam2world.txt')

    # open video
    cap = cv2.VideoCapture(os.path.join('datasets', 'TownCentreXVID.avi'))

    i_frame = 0
    while cap.isOpened() and i_frame < 5000:

        ret, img = cap.read()
        if not i_frame % 10 == 0:
            i_frame += 1
            continue

        t0 = time.time()
        img_t = np.moveaxis(img, -1, 0) / 255
        img_t = torch.tensor(img_t, device=device).float()
        predictions = model([img_t])

        boxes = predictions[0]['boxes'].cpu().data.numpy()
        classIDs = predictions[0]['labels'].cpu().data.numpy()
        scores = predictions[0]['scores'].cpu().data.numpy()

        pos_world = []

        for i in range(len(boxes)):
            if classIDs[i] == 1 and scores[i] > 0.9:
                # extract the bounding box coordinates
                (y1, x1) = (boxes[i][0], boxes[i][1])
                (y2, x2) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                color = [0, 0, 255]
                cv2.rectangle(img, (y1, x1), (y2, x2), color, 2)

                text = "{}: {:.4f}".format(LABELS[classIDs[i]], scores[i])
                cv2.putText(img, text, (int(y1), int(x1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # find the position
                p_c = np.array([[(y1 + y2)/2], [x2], [1]])
                p_w = M_c2w @ p_c
                p_w = p_w / p_w[2]
                pos_world.append([p_w[0][0], p_w[1][0]])

        pos_world = np.array(pos_world)
        violation_pairs = find_violation(pos_world)
        pos_world_vis = pos_world * 10

        # plot
        fig = plt.figure(figsize=(10, 10))
        # subplot 1 - camera view
        a = fig.add_subplot(2, 1, 1)
        plt.imshow(img)
        a.set_title('Video')
        # subplot 2 - bird eye view background
        a = fig.add_subplot(2, 2, 3)
        plt.imshow(img_bkgd_bev)
        a.set_title('BEV')
        a.plot(pos_world_vis[:, 1], pos_world_vis[:, 0], 'or', alpha=0.5)
        a.axis('equal')
        a.grid()
        a.set(xlim=(300, -100), ylim=(400, 0))
        a.invert_yaxis()
        a.set_xlabel('0.1m / pixel')
        a.set_ylabel('0.1m / pixel')

        # subplot 3 - bird eye view social distancing
        a = fig.add_subplot(2, 2, 4)
        a.set_title('BEV - social distancing')
        a.plot(pos_world[:, 1], pos_world[:, 0], 'or', alpha=0.5)
        for pair in violation_pairs:
            data = np.array([pos_world[pair[0]], pos_world[pair[1]]])
            a.plot(data[:, 1], data[:, 0], '-b')
        a.axis('equal')
        a.grid()
        a.set(xlim=(-20, 20), ylim=(0, 40))
        a.invert_xaxis()
        a.set_xlabel('meters')
        a.set_ylabel('meters')


        fig.savefig('results/frame%04d.png' % i_frame)
        plt.close(fig)

        i_frame += 1

        t1 = time.time()
        print('Processing Time: %.2f' % (t1 - t0))

        print('=======================')



if __name__ == '__main__':
    main()