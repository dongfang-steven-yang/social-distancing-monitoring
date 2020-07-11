import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

    meter_per_pixel = 0.4 / 30

    # pts_image = np.float32([[86, 315], [125, 247], [502, 236], [529, 299]])

    pts_image = np.float32([[87, 314], [125, 247], [502, 236], [529, 299]])
    pts_world = np.float32([[0, 443.29/2], [0, 0], [443.29, 0], [443.29, 443.29/2]]) * meter_per_pixel
    pts_world_10x = pts_world * 10

    matrix_cam2world = cv2.getPerspectiveTransform(pts_image, pts_world)
    matrix_cam2world10x = cv2.getPerspectiveTransform(pts_image, pts_world_10x)

    # save transformation matrix
    np.savetxt('mall_matrix_cam2world.txt', matrix_cam2world)
    print('Matrix Saved.')

    # background in bev
    img = cv2.imread('mall_background.png')
    dst = cv2.warpPerspective(img, matrix_cam2world10x, (100, 100))
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()
    cv2.imwrite('mall_background_calibrated.png', dst)
    print('Converted background saved.')


if __name__ == '__main__':
    main()

