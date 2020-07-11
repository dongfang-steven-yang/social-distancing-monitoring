import numpy as np
from scipy import stats


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

ROIs = {
    'mall': (-2., 8., -7., 7.),
    'grand_central': (-7., 25., 0., 34.),
    'oxford_town':  (0., 14., 5., 28.),
}


def decode_data(data, roi):
    """
    - decode the raw data w.r.t. the defined roi.

    :param data:
    :param roi:
    :return:
    """
    x_min, x_max, y_min, y_max = roi
    area = (x_max - x_min) * (y_max - y_min)

    density = []
    ts_inference = []
    pts_roi_all_frame = []
    inds_frame = []
    nums_ped = []

    for i_frame, t_inference, pts in data:
        count_in = 0
        count_out = 0
        pts_roi = []
        for pt in pts:
            if x_min < pt[0] < x_max and y_min < pt[1] < y_max:
                count_in += 1
                pts_roi.append(pt)
            else:
                count_out += 1
        pts_roi_all_frame.append(np.array(pts_roi))
        density.append(count_in / area)
        ts_inference.append(t_inference)
        inds_frame.append(i_frame)
        nums_ped.append(count_in)

        # print('frame %d - num. of ped inside roi: %d, outside: %d' % (i_frame, count_in, count_out))

    return np.array(inds_frame), np.array(ts_inference), pts_roi_all_frame, np.array(density), nums_ped


def count_violation_pairs(pts_all_frames, dist=2.0):
    counts = []
    for pts in pts_all_frames:
        pairs = find_violation(pts, dist)
        counts.append(len(pairs))
    return np.array(counts)


def find_violation(pts, dist=2.0):
    """

    :param pts: positions of all pedestrians in a single frame
    :param dist: social distance
    :return: a list of index pairs indicating two pedestrians who are violating social distancing
    """
    n = len(pts)  # number of pedestrians
    pairs = []
    for i in np.arange(0, n, 1):
        for j in np.arange(i+1, n, 1):
            if np.linalg.norm(pts[i] - pts[j]) < dist:
                pairs.append((i, j))
    return pairs


def cal_min_dists_all_frame(pts_all_frame):
    all_min_dists = []
    avg_min_dists = []
    min_min_dists = []
    for pts in pts_all_frame:
        min_dists = cal_min_dists(pts)
        all_min_dists.append(min_dists)
        min_min_dists.append(min(min_dists) if len(min_dists) > 0 else None)
        avg_min_dists.append(sum(min_dists) / len(min_dists) if len(min_dists) > 0 else None)

    all_min_dists = sum(all_min_dists, [])

    return all_min_dists, np.array(min_min_dists), np.array(avg_min_dists)


def cal_min_dists(pts):
    """

    :param pts: positions of all pedestrians in a single frame
    :return: a list of each pedestrian's min distances to other pedestrians
    """
    n = len(pts)
    ds_min = []
    for i in range(n):
        d_min = np.inf
        for j in range(n):
            # closest distance from pedestrian i to pedestrian j
            if i != j and np.linalg.norm(pts[i] - pts[j]) < d_min:
                d_min = np.linalg.norm(pts[i] - pts[j])
        if d_min is not np.inf:
            ds_min.append(d_min)
    return ds_min


def custom_simple_linear_regression(xs, ys, x_select):

    # reference 1: http://www2.stat.duke.edu/~tjl13/s101/slides/unit6lec3H.pdf
    # reference 2: http://statweb.stanford.edu/~susan/courses/s141/horegconf.pdf

    # assume: y = a + b*x + e

    def pred_interval(x_star, prob=0.95):
        se_pred = np.sqrt(variance * (1 + 1 / n + (x_star - xs.mean()) ** 2 / s_xx))
        t_dist = stats.t.ppf([1 - (1 - prob)/2], len(xs) - 2)
        y_hat = x_star * b + a
        return float(y_hat - se_pred * t_dist), float(y_hat + se_pred * t_dist)

    b, a, r_value, p_value, std_err = stats.linregress(xs, ys)

    print('slope b = %.6f' % b)
    print('intercept a = %.6f' % a)
    print('r_value = %.6f' % r_value)
    print('p_value = %.6f' % p_value)
    print('se_a (from package) = %.6f' % std_err)

    residuals = ys - (xs * b + a)
    n = len(residuals)
    variance = np.sum(residuals ** 2) / (n - 2)
    s_xx = np.sum((xs - xs.mean()) ** 2)
    s_x = s_xx / (n - 1)

    se_a = np.sqrt(variance / s_xx)
    print('se_a_ = %.6f' % se_a)

    # se_intercept_wiki = std_err * np.sqrt(np.sum(xs ** 2) / n)
    # print('se_intercept_wiki = %.6f' % se_intercept_wiki)

    preds, lbs, ubs = [], [], []  # prediction interval
    for x in np.arange(0, np.max(xs), 0.001):
        lb, ub = pred_interval(x)
        preds.append(x)
        lbs.append(lb)
        ubs.append(ub)
    if x_select == 'x_intercept':
        x_select = - a / b
    elif x_select == 'y_intercept':
        x_select = 0.0
    else:
        raise Exception
    lb_select, ub_select = pred_interval(x_select)

    return a, b, preds, lbs, ubs, x_select, lb_select, ub_select

