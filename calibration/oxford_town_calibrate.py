import numpy as np
from scipy.spatial.transform import Rotation
import cv2


def projection_matrices_oxford_town():

    # intrinsic matrix
    F_X = 2696.35888671875000000000
    F_Y = 2696.35888671875000000000
    C_X = 959.50000000000000000000
    C_Y = 539.50000000000000000000
    intrinsic_matrix = np.array([
        [F_X, 0, C_X],
        [0, F_Y, C_Y],
        [0, 0, 1]
    ])

    # extrinsic matrix
    rotation_quaternion = [
        0.69724917918208628720,
        -0.43029624469563848566,
        0.28876888503799524877,
        0.49527896681027261394
    ]
    r = Rotation.from_quat(rotation_quaternion)
    rotation_matrix = r.as_matrix()
    translation = np.array([[-0.05988363921642303467], [3.83331298828125000000], [12.39112186431884765625]])
    extrinsic_matrix = np.concatenate((rotation_matrix, translation), axis=1)

    return intrinsic_matrix, extrinsic_matrix


def project_w2c(p, in_mat, ex_mat, distortion=False):
    # extrinsic
    P = np.array(p).reshape(4, 1)
    p_temp = ex_mat @ P

    # distortion
    if distortion:
        K1 = -0.60150605440139770508
        K2 = 4.70203733444213867188
        P1 = -0.00047452122089453042
        P2 = -0.00782289821654558182
        x_p = p_temp[0][0]
        y_p = p_temp[1][0]
        r_sq = x_p ** 2 + y_p ** 2
        xpp = x_p * (1 + K1 * r_sq + K2 * (r_sq ** 2)) + 2 * P1 * x_p * y_p + P2 * (r_sq + 2 * (x_p ** 2))
        ypp = y_p * (1 + K1 * r_sq + K2 * (r_sq ** 2)) + 2 * P2 * x_p * y_p + P1 * (r_sq + 2 * (y_p ** 2))
        p_temp[0][0] = xpp
        p_temp[1][0] = ypp

    # intrinsic
    p = in_mat @ p_temp
    p = p / p[2]

    return np.int(p[0]), np.int(p[1])


# def visualize_camera_cal_oxford_town():
#     os.chdir('..')
#     os.makedirs(cfg['path']['vis'], exist_ok=True)
#     cap = cv2.VideoCapture(os.path.join(cfg['path']['data'], "TownCentreXVID.avi"))
#     if cap.isOpened():
#         ret, img = cap.read()
#         HEIGHT = img.shape[0]
#         WIDTH = img.shape[1]
#         DEPTH = img.shape[2]
#         h_cal = 1000  # 20 pixels = 1 meter
#         w_cal = 1000
#         img_cal = np.zeros((h_cal, w_cal, 3))
#         in_mat, ex_mat = projection_matrices_oxford_town()
#         for i in range(h_cal):
#             for j in range(w_cal):
#                 x, y = project_w2c([i / 20, j / 20, 0, 1], in_mat, ex_mat)
#                 print(x, y)
#                 if 0 <= y < HEIGHT and 0 <= x < WIDTH:
#                     img_cal[i, j, :] = img[y, x, :]
#
#         cv2.imwrite(os.path.join(cfg['path']['vis'], 'oxford_original.png'), img)
#         cv2.imwrite(os.path.join(cfg['path']['vis'], 'oxford_calibrated.png'), img_cal)
#         print('completed.')


def convert_background(multiplier=10):
    img = cv2.imread('oxford_town_background.png')
    HEIGHT = img.shape[0]
    WIDTH = img.shape[1]
    DEPTH = img.shape[2]
    h_cal = 50 * multiplier  # 10 pixels = 1 meter
    w_cal = 50 * multiplier
    img_cal = np.zeros((h_cal, w_cal, 3))
    in_mat, ex_mat = projection_matrices_oxford_town()
    for i in range(h_cal):
        for j in range(w_cal):
            x, y = project_w2c([i / multiplier, j / multiplier, 0, 1], in_mat, ex_mat)
            if 0 <= y < HEIGHT and 0 <= x < WIDTH:
                img_cal[i, j, :] = img[y, x, :]

    cv2.imwrite('oxford_town_background_calibrated.png', img_cal)
    print('Calibrated background saved.')


def save_transforamtion_matrix():
    in_mat, ex_mat = projection_matrices_oxford_town()
    M_c2w = np.linalg.inv(in_mat @ np.delete(ex_mat, 2, axis=1))
    np.savetxt('oxford_town_matrix_cam2world.txt', M_c2w)

    print('Matrix saved.')



if __name__ == '__main__':
    # visualize_camera_cal_oxford_town()
    convert_background()
    save_transforamtion_matrix()
