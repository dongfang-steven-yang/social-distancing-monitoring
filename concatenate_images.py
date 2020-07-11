from PIL import Image
import os
import glob

# result_folder = 'additional_detectors/yolo_v4_darknet_official/results/result_yolo_v4_darknet_official'
result_folder = 'results/20200624_125554/analysis'  # faster RCNN results


def get_concat_h(im1, im2, im3):
    dst = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.paste(im3, (im1.width + im2.width, 0))

    return dst


def get_concat_v(im1, im2, im3):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height + im3.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    dst.paste(im3, (0, im1.height + im1.height))

    return dst


fs = ['hist_density_vs_avg_dists', 'hist_density_vs_violation', 'regression_density_vs_violation']

for f in fs:
    files = glob.glob(os.path.join(result_folder, '*%s.png' % f))
    files = sorted(files, reverse=True)
    assert len(files) == 3
    im1 = Image.open(files[0])
    im2 = Image.open(files[1])
    im3 = Image.open(files[2])

    os.makedirs(os.path.join(result_folder, 'combined'), exist_ok=True)
    get_concat_h(im1, im2, im3).save(os.path.join(result_folder, 'combined', '%s.png' % f))

fs = ['statistics_vs_time']

for f in fs:
    files = glob.glob(os.path.join(result_folder, '*%s.png' % f))
    files = sorted(files, reverse=True)
    assert len(files) == 3
    im1 = Image.open(files[0])
    im2 = Image.open(files[1])
    im3 = Image.open(files[2])

    os.makedirs(os.path.join(result_folder, 'combined'), exist_ok=True)
    get_concat_v(im1, im2, im3).save(os.path.join(result_folder, 'combined', '%s.png' % f))


files = glob.glob(os.path.join(result_folder, 'demo', '*.png'))
files = sorted(files, reverse=True)
assert len(files) == 3
im1 = Image.open(files[0])
im2 = Image.open(files[1])
im3 = Image.open(files[2])

os.makedirs(os.path.join(result_folder, 'combined'), exist_ok=True)
get_concat_v(im1, im2, im3).save(os.path.join(result_folder, 'combined', 'detection.png' ))