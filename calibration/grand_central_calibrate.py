import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

    meter_per_pixel = 4.8768 / 144

    pts_image = np.float32([[248, 100], [435, 100], [473, 287], [202, 287]])
    pts_world = np.float32([[0, 0], [440, 0], [440, 672], [0, 672]]) * meter_per_pixel

    pts_world_10x = pts_world * 10

    matrix_cam2world = cv2.getPerspectiveTransform(pts_image, pts_world)
    matrix_cam2world10x = cv2.getPerspectiveTransform(pts_image, pts_world_10x)

    # save transformation matrix
    np.savetxt('grand_central_matrix_cam2world.txt', matrix_cam2world)
    print('Matrix Saved.')

    # background in bev
    img = cv2.imread('grand_central_background.png')
    dst = cv2.warpPerspective(img, matrix_cam2world10x, (400, 400))
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()
    cv2.imwrite('grand_central_background_calibrated.png', dst)
    print('Converted background saved.')



if __name__ == '__main__':
    main()

