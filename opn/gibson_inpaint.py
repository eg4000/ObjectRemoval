# -*- coding: utf-8 -*-

from __future__ import division

import ast
import itertools
import re

import torch
import yaml
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F

from GibsonEnv.gibson.data.datasets import get_model_path
from GibsonEnv.gibson.envs.drone_env import DroneNavigateEnv

# general libs
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import math
import time
import os
import sys
import argparse
import glob
import shutil

### My libs
sys.path.append('utils/')
sys.path.append('models/')
from opn.utils.helpers import *
from opn.models.OPN import OPN


def get_arguments():
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument('--config', type=str, default=config_file)

    return parser.parse_args()


def get_color_mask(img, obj_color):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_img = np.zeros((1, 1, 3), np.uint8)
    color_img[:] = obj_color
    obj_color_hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)[0, 0]

    h_tol = 10
    s_tol = 10
    v_tol = 10
    lower_limit = np.array([obj_color_hsv[0] - h_tol, min(obj_color_hsv[1] - s_tol, 255), min(obj_color_hsv[2] - v_tol, 255)])
    upper_limit = np.array([obj_color_hsv[0] + h_tol, max(obj_color_hsv[1] + s_tol, 0), max(obj_color_hsv[2] + v_tol, 0)])

    if lower_limit[0] < 0 or upper_limit[0] > 179:
        lower_limit1 = np.array([0, lower_limit[1], lower_limit[2]])
        upper_limit1 = np.array([h_tol, upper_limit[1], upper_limit[2]])
        lower_limit2 = np.array([170, lower_limit[1], lower_limit[2]])
        upper_limit2 = np.array([180, upper_limit[1], upper_limit[2]])
        mask1 = cv2.inRange(hsv, lower_limit1, upper_limit1)
        mask2 = cv2.inRange(hsv, lower_limit2, upper_limit2)
        mask = mask1 + mask2

    else:
        mask = cv2.inRange(hsv, lower_limit, upper_limit)
    return mask


def get_mask(img):
    mask1 = get_color_mask(img, obj_color)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    # sure background area
    sure_bg = cv2.dilate(mask1, kernel, iterations=3)
    contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(sure_bg, [cnt], 0, 255, -1)

    kernel = np.ones((5, 5), np.uint8)
    sure_bg = cv2.erode(sure_bg, kernel, iterations=3)
    mask2 = cv2.dilate(sure_bg, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for ii in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[ii], False))
    # create an empty black image
    final_mask = np.zeros_like(mask2)

    # draw contours and hull points
    for ii in range(len(contours)):
        cv2.drawContours(final_mask, hull, ii, 255, -1)
    return final_mask


if __name__ == '__main__':

    ####################    Navigate    ####################

    ####################Load env
    offset = [1.66530786, 0.6650605, 0.05189579]
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'play_drone_camera.yaml')
    args = get_arguments()
    with open(args.config, 'r') as f:
        config_data = yaml.load(f)
    # config_data['model_id'] = 'Collierville'
    config_data['initial_pos'] = [0, 0, 0]
    config_data['initial_orn'] = [0, 0, 3.14]
    env = DroneNavigateEnv(config=config_data)
    env2 = DroneNavigateEnv(config=config_data, pano_type='masks')
    env.reset()
    env2.reset()
    model_id = config_data['model_id']
    model_path = get_model_path(model_id)
    attribute_file = os.path.join(model_path, 'attributes.csv')
    attribute_df = pd.read_csv(attribute_file, index_col=0, header=None).T
    for row_i, obj in attribute_df.iterrows():
        #################### For each object
        location = ast.literal_eval(re.sub(r"\s+", ',', obj['location']).replace('[,', '[').replace(',]', ']'))
        # location = obj['location']
        obj_color = ast.literal_eval(re.sub(r"\s+", ',', obj['color']).replace('[,', '[').replace(',]', ']'))
        # obj_color = obj['color']
        # b, g, r = obj_color
        # obj_color = np.asarray([r, g, b])
        class_ = obj['class_']
        if class_ not in ['chair', 'bench', 'couch', 'dining table']:
        # if class_ not in ['couch']:
            continue
        initial_pos = [location[0] + offset[0], location[1] + offset[1], location[2] + offset[2]]
        seq_name = class_ + obj.id
        print(seq_name)
        folder = os.path.join('Image_inputs', model_id, seq_name)

        if os.path.isdir(folder):
            shutil.rmtree(os.path.join(folder))
        os.makedirs(folder)

        save_path = os.path.join('Image_results', model_id, seq_name)
        if os.path.isdir(save_path):
            shutil.rmtree(os.path.join(save_path))
        os.makedirs(save_path)

        #################### Render and save trajectory
        i = 0

        eye_pos, eye_quat = env.get_eye_pos_orientation()
        orig_pose = [initial_pos, eye_quat]
        pose = [np.copy(orig_pose[0]), np.copy(orig_pose[1])]

        # color_img = np.zeros((50, 50, 3), np.uint8)
        # color_img[:] = obj_color
        # cv2.imwrite(os.path.join(folder, '0.jpeg'), cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

        # pose[0][0] += x;cv2.imwrite('{}/{:05}.jpg'.format(folder, i), cv2.cvtColor(env.render_observations(pose)['rgb_filled'], cv2.COLOR_BGR2RGB));i=+1


        for x in np.arange(0, 0.32, 0.05):
            pose[0][2] += x
            canvas = Image.fromarray(env.render_observations(pose)['rgb_filled'])
            canvas.save(os.path.join(folder, '{:05}.jpg'.format(i)))
            canvas = Image.fromarray(get_mask((env2.render_observations(pose)['rgb_prefilled'])))
            canvas.save(os.path.join(folder, '{:05}.png'.format(i)))
            i += 1
        for x in np.arange(0, 0.46, 0.05):
            pose[0][0] -= x
            canvas = Image.fromarray(env.render_observations(pose)['rgb_filled'])
            canvas.save(os.path.join(folder, '{:05}.jpg'.format(i)))
            canvas = Image.fromarray(get_mask((env2.render_observations(pose)['rgb_prefilled'])))
            canvas.save(os.path.join(folder, '{:05}.png'.format(i)))
            i += 1
        for x in np.arange(0, 0.27, 0.05):
            pose[0][2] -= x
            canvas = Image.fromarray(env.render_observations(pose)['rgb_filled'])
            canvas.save(os.path.join(folder, '{:05}.jpg'.format(i)))
            canvas = Image.fromarray(get_mask((env2.render_observations(pose)['rgb_prefilled'])))
            canvas.save(os.path.join(folder, '{:05}.png'.format(i)))
            i += 1

        ####################    Inpaint    ####################

        #################### Load image
        # T, H, W = 5, 240, 424
        H, W = 256, 256
        input_images = glob.glob(os.path.join(folder, '*.jpg'))
        input_masks = glob.glob(os.path.join(folder, '*.png'))
        input_images.sort()
        input_masks.sort()
        assert (len(input_images) == len(input_masks))
        T = len(input_masks)
        print('{} frames'.format(T))
        frames = np.empty((T, H, W, 3), dtype=np.float32)
        holes = np.empty((T, H, W, 1), dtype=np.float32)
        dists = np.empty((T, H, W, 1), dtype=np.float32)

        for i, (img_file, mask_file) in enumerate(zip(input_images, input_masks)):
            #### rgb
            raw_frame = np.array(Image.open(img_file).convert('RGB')) / 255.
            raw_frame = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
            frames[i] = raw_frame
            #### mask
            raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            raw_mask = (raw_mask > 0.5).astype(np.uint8)
            raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
            holes[i, :, :, 0] = raw_mask.astype(np.float32)
            #### dist
            dists[i, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2, maskSize=5)

        frames = torch.from_numpy(np.transpose(frames, (3, 0, 1, 2)).copy()).float()
        holes = torch.from_numpy(np.transpose(holes, (3, 0, 1, 2)).copy()).float()
        dists = torch.from_numpy(np.transpose(dists, (3, 0, 1, 2)).copy()).float()
        # remove hole
        frames = frames * (1 - holes) + holes * torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        # valids area
        valids = 1 - holes
        # unsqueeze to batch 1
        frames = frames.unsqueeze(0)
        holes = holes.unsqueeze(0)
        dists = dists.unsqueeze(0)
        valids = valids.unsqueeze(0)

        #################### Load Model
        model = nn.DataParallel(OPN())
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(os.path.join('OPN.pth')), strict=False)
        model.eval()

        ################### Inference
        # memory encoding
        # frame_skips = list(range(2, int(math.ceil(((T + 1))))))
        # frame_sublist = list(powerset(list(range(1, T+1))))
        ref_frames = list(range(1, T))
        # ref_frames = [f for f in list(range(1, T)) if f % 3 != 0 and f % 5 != 0]
        # print(ref_frames)
        # frame_sublists = list(itertools.combinations(ref_frames, 5))
        frame_sublists = [[1, 2, 3, 4, 5, 17], [1, 2, 3, 4, 7, 15, 18], [1, 2, 3, 4, 6, 20]]
        for midx in frame_sublists:
            # midx = list(range(0, T, frame_skip))
            print(midx)
            with torch.no_grad():
                mkey, mval, mhol = model(frames[:, :, midx], valids[:, :, midx], dists[:, :, midx])

            # for f in range(T):
            for f in [0]:
                # memory selection
                # print('memory selection {}/{}'.format(f, T))

                ridx = [i for i in range(len(midx)) if i != f]  # memory minus self
                fkey, fval, fhol = mkey[:, :, ridx], mval[:, :, ridx], mhol[:, :, ridx]
                # inpainting..
                for r in range(999):
                    if r == 0:
                        comp = frames[:, :, f]
                        dist = dists[:, :, f]
                    with torch.no_grad():
                        comp, dist = model(fkey, fval, fhol, comp, valids[:, :, f], dist)

                    # update
                    comp, dist = comp.detach(), dist.detach()
                    if torch.sum(dist).item() == 0:
                        break
                # visualize..
                est = (comp[0].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)
                true = (frames[0, :, f].permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)  # h,w,3
                mask = (dists[0, 0, f].detach().cpu().numpy() > 0).astype(np.uint8)  # h,w,1
                ov_true = overlay_davis(true, mask, colors=[[0, 0, 0], [100, 100, 0]], cscale=2, alpha=0.4)

                canvas = np.concatenate([ov_true, est], axis=0)

                canvas = Image.fromarray(canvas)
                canvas.save(os.path.join(save_path, 'res_{}.jpg'.format(midx)))

        print('Results are saved: ./{}'.format(save_path))
