import cv2
import numpy as np
from calibration.background import generate_background

input_video = '../datasets/grandcentral.avi'
output_background = 'grand_central_background.png'


def method1():
    cap = cv2.VideoCapture('')
    _, f = cap.read()
    img_bkgd = np.float32(f)
    print('When you feel the background is good enough, press ESC to terminate and save the background.')

    while True:
        _, f = cap.read()

        cv2.accumulateWeighted(f, img_bkgd, 0.01)
        res2 = cv2.convertScaleAbs(img_bkgd)

        # cv2.imshow('img', f)
        cv2.imshow('When you feel the background is good enough, press ESC to terminate and save the background.', res2)
        k = cv2.waitKey(20)

        if k == 27:
            break

    cv2.imwrite(output_background, res2)
    cv2.destroyAllWindows()
    cap.release()


def method2():
    img_bkgd = generate_background(path_video=input_video)
    cv2.imwrite(output_background, img_bkgd)

if __name__ == '__main__':
    # method1()
    method2()