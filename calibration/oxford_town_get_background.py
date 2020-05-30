import cv2
import numpy as np

cap = cv2.VideoCapture('../datasets/TownCentreXVID.avi')
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

cv2.imwrite('oxford_town_background.png', res2)
cv2.destroyAllWindows()
cap.release()
