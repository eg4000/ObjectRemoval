import glob
import math
import os
import numpy as np
import cv2


def avg(lst):
    return sum(lst) / len(lst)


def eval(x, y):
    x = x / 255
    y = y / 255

    l1_loss = np.sum(np.abs(x - y)) / mask_area / 3.
    l2_loss = np.sqrt(np.sum(np.square(x - y))) / mask_area / 3.
    psnr_loss = -10.0 * math.log10(np.sum(np.square(x - y)) / mask_area / 3.)
    img = np.multiply(x, mask)
    pixel_dif1 = img[1:, :, :] - img[:-1, :, :]
    pixel_dif2 = img[:, 1:, :] - img[:, :-1, :]
    tv_loss = (np.sum(np.abs(pixel_dif1)) + np.sum(np.abs(pixel_dif2))) / np.sum(mask)

    return l1_loss, l2_loss, psnr_loss, tv_loss


if __name__ == '__main__':
    gt_path = 'generative_inpainting/examples/val_256/'
    SN_PatchGAN_prediction_path = 'generative_inpainting/examples/val_256_output/'
    opn_prediction_path = 'opn/Image_results/val_256/'
    gt_images = glob.glob(os.path.join(gt_path, '*.jpg'))
    SN_PatchGAN_predictions = set(glob.glob(os.path.join(SN_PatchGAN_prediction_path, '*.jpg')))
    opn_predictions = glob.glob(os.path.join(opn_prediction_path, 'est_*.png'))
    opn_predictions = [name.replace('est_', '') for name in opn_predictions]

    opn_predictions = set([name.split('/')[-1][:-4] for name in opn_predictions])
    gt_images = set([name.split('/')[-1][:-4] for name in gt_images])
    SN_PatchGAN_predictions = set([name.split('/')[-1][:-4] for name in SN_PatchGAN_predictions])

    print(len(gt_images), len(SN_PatchGAN_predictions), len(opn_predictions))
    gt_images = gt_images.intersection(opn_predictions)
    gt_images = gt_images.intersection(SN_PatchGAN_predictions)
    SN_PatchGAN_predictions = gt_images.intersection(opn_predictions)
    opn_predictions = gt_images.intersection(SN_PatchGAN_predictions)
    assert opn_predictions == SN_PatchGAN_predictions
    assert gt_images == SN_PatchGAN_predictions
    gt_images = sorted(gt_images)
    SN_PatchGAN_predictions = sorted(SN_PatchGAN_predictions)
    opn_predictions = sorted(opn_predictions)

    mask = cv2.imread('generative_inpainting/examples/mask.png')
    mask = mask / 255
    mask_area = mask.sum() / 3
    img_area = mask.shape[0] * mask.shape[1]
    mask_height, mask_width = np.sqrt(mask_area), np.sqrt(mask_area)

    l1_sns = []
    l2_sns = []
    psnr_sns = []
    tv_sns = []

    l1_opns = []
    l2_opns = []
    psnr_opns = []
    tv_opns = []

    total = len(gt_images)
    for i, (gt, sn_patchGAN, opn) in enumerate(zip(gt_images, SN_PatchGAN_predictions, opn_predictions)):
        print('{}/{}'.format(i, total))
        y = cv2.imread(gt_path + gt + '.jpg')
        x_sn = cv2.imread(SN_PatchGAN_prediction_path + sn_patchGAN + '.jpg')
        x_opn = cv2.imread(opn_prediction_path + 'est_' + opn + '.png')

        l1_sn, l2_sn, psnr_sn, tv_sn = eval(x_sn, y)
        l1_opn, l2_opn, psnr_opn, tv_opn = eval(x_opn, y)

        l1_sns.append(l1_sn)
        l2_sns.append(l2_sn)
        psnr_sns.append(psnr_sn)
        tv_sns.append(tv_sn)

        l1_opns.append(l1_opn)
        l2_opns.append(l2_opn)
        psnr_opns.append(psnr_opns)
        tv_opns.append(tv_opn)

    print('l1: sn={}, opn={}'.format(avg(l1_sns), avg(l1_opns)))
    print('l2: sn={}, opn={}'.format(avg(l2_sns), avg(l2_opns)))
    print('psnr: sn={}, opn={}'.format(avg(psnr_sns), avg(psnr_opns)))
    print('tv: sn={}, opn={}'.format(avg(tv_sns), avg(tv_opns)))
