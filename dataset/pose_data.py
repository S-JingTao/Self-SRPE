#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Jingtao Sun
# @Filename    : pose_data.py
import os
import cv2
import copy

import numpy as np
import _pickle as cPickle
import torch.utils.data as data
import torchvision.transforms as transforms

from utils.dataset_utils import load_depth, get_2dbbox, read_meta_file, read_meta_file_camera


class PoseDataset(data.Dataset):
    def __init__(self, source, mode, data_dir, n_pts):
        """
        :param source: ‘CAMERA’, 'REAL', 'CAMERA+REAL', 'DYNAMIC'
        :param mode: 'train', 'test'
        :param data_dir: dataset save_path
        :param n_pts: point num
        """

        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts

        assert source in ['CAMERA', 'REAL', 'CAMERA+Real', 'DYNAMIC']
        assert mode in ['train', 'test']

        data_list_path = ['CAMERA/train_list_all.txt', 'Real/train_list_all.txt',
                          'CAMERA/val_list_all.txt', 'Real/test_list_all.txt']

        model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
                           'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']

        cur_data_list = []
        cur_model_file = []

        if source == 'CAMERA':
            if mode == "train":
                cur_data_list.append(data_list_path[0])
                cur_model_file.append(model_file_path[0])
            else:
                cur_data_list.append(data_list_path[2])
                cur_model_file.append(model_file_path[2])
        elif source == 'REAL':
            if mode == 'train':
                cur_data_list.append(data_list_path[1])
                cur_model_file.append(model_file_path[1])
            else:
                cur_data_list.append(data_list_path[3])
                cur_model_file.append(model_file_path[3])

        img_list = []
        subset_len = []

        # load img_path_list
        for path in cur_data_list:
            img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(data_dir, path))]
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1] - subset_len[0]]

        self.img_list = img_list
        self.length = len(self.img_list)

        # load_model
        models = {}
        for path in cur_model_file:
            with open(os.path.join(data_dir, path), 'rb') as f:
                models.update(cPickle.load(f))

        self.models = models

        # meta info for re-label mug category
        with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        self.mean_shapes = np.load("/home/sjt/Categorical-3D-SRPE/dataset/mean_points_emb.npy")
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.camera_intrinsics = [577.5, 577.5, 319.5, 239.5]  # [fx, fy, cx, cy]
        self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        self.sym_ids = [0, 1, 3]  # 0-indexed
        self.norm_scale = 1000.0  # normalization scale
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.shift_range = 0.01
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        print('{} images found.'.format(self.length))
        print('{} models loaded.'.format(len(self.models)))

    def enlarge_bbox(self, target):

        def _search_fit(points):
            min_x = min(points[:, 0])
            max_x = max(points[:, 0])
            min_y = min(points[:, 1])
            max_y = max(points[:, 1])
            min_z = min(points[:, 2])
            max_z = max(points[:, 2])

            return [min_x, max_x, min_y, max_y, min_z, max_z]

        limit = np.array(_search_fit(target))
        longest = max(limit[1] - limit[0], limit[3] - limit[2], limit[5] - limit[4])
        longest = longest * 1.3

        scale1 = longest / (limit[1] - limit[0])
        scale2 = longest / (limit[3] - limit[2])
        scale3 = longest / (limit[5] - limit[4])

        target[:, 0] *= scale1
        target[:, 1] *= scale2
        target[:, 2] *= scale3

        return target

    def get_pose_camera(self, img_num, cur_idx):
        has_pose = []
        pose = {}
        if self.mode == "train":
            cur_frame_name = os.path.join(self.data_dir, 'gts', 'train', img_num[0], img_num[1])
            input_file = open('{0}_pose.txt'.format(cur_frame_name), 'r')
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                if len(input_line) == 1:
                    idx = int(input_line[0])
                    has_pose.append(idx)
                    pose[idx] = []
                    for i in range(4):
                        input_line = input_file.readline()
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        input_line = input_line.split(' ')
                        pose[idx].append(
                            [float(input_line[0]), float(input_line[1]), float(input_line[2]), float(input_line[3])])
            input_file.close()

        if self.mode == "test":
            with open('{0}/gts/val/results_val_{1}_{2}.pkl'.format(self.data_dir, img_num[0],
                                                                   img_num[1]),
                      'rb') as f:
                nocs_data = cPickle.load(f)
            for idx in range(nocs_data['gt_RTs'].shape[0]):
                idx = idx + 1
                pose[idx] = nocs_data['gt_RTs'][idx - 1]
                pose[idx][:3, :3] = pose[idx][:3, :3] / np.cbrt(np.linalg.det(pose[idx][:3, :3]))
                z_180_RT = np.zeros((4, 4), dtype=np.float32)
                z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                z_180_RT[3, 3] = 1
                pose[idx] = z_180_RT @ pose[idx]
                pose[idx][:3, 3] = pose[idx][:3, 3] * 1000

        if int(cur_idx) not in pose.keys():
            return 0, 0
        ans = pose[int(cur_idx)]

        ans = np.array(ans)
        ans_r = ans[:3, :3]
        ans_t = ans[:3, 3].flatten()

        return ans_r, ans_t

    def get_bbox_camera(self, file_path):
        has_bbox = []
        bbox = {}
        input_file = open(file_path, 'r')

        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            if len(input_line) == 1:
                idx = int(input_line[0])
                has_bbox.append(idx)
                bbox[idx] = []
                for i in range(4):
                    input_line = input_file.readline()
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                    input_line = input_line.split(' ')
                    bbox[idx].append(
                        [float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        return bbox

    def get_pose_real(self, img_num, cur_idx):

        has_pose = []
        pose = {}
        if self.mode == "train":
            cur_frame_name = os.path.join(self.data_dir, 'gts', 'real_train', img_num[0], img_num[1])
            input_file = open('{0}_pose.txt'.format(cur_frame_name), 'r')
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                if len(input_line) == 1:
                    idx = int(input_line[0])
                    has_pose.append(idx)
                    pose[idx] = []
                    for i in range(4):
                        input_line = input_file.readline()
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        input_line = input_line.split(' ')
                        pose[idx].append(
                            [float(input_line[0]), float(input_line[1]), float(input_line[2]), float(input_line[3])])
            input_file.close()

        if self.mode == "test":
            with open('{0}/gts/real_test/results_real_test_{1}_{2}.pkl'.format(self.data_dir, img_num[0],
                                                                               img_num[1]),
                      'rb') as f:
                nocs_data = cPickle.load(f)
            for idx in range(nocs_data['gt_RTs'].shape[0]):
                idx = idx + 1
                pose[idx] = nocs_data['gt_RTs'][idx - 1]
                pose[idx][:3, :3] = pose[idx][:3, :3] / np.cbrt(np.linalg.det(pose[idx][:3, :3]))
                z_180_RT = np.zeros((4, 4), dtype=np.float32)
                z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                z_180_RT[3, 3] = 1
                pose[idx] = z_180_RT @ pose[idx]
                pose[idx][:3, 3] = pose[idx][:3, 3] * 1000

        ans = pose[int(cur_idx)]

        ans = np.array(ans)
        ans_r = ans[:3, :3]
        ans_t = ans[:3, 3].flatten()

        return ans_r, ans_t

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.img_list[item])
        # rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        depth = load_depth(img_path + "_depth.png")
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2]
        coord = cv2.imread(img_path + '_coord.png')[:, :, :3]
        coord = coord[:, :, (2, 1, 0)]
        coord = np.array(coord, dtype=np.float32) / 255
        coord[:, :, 2] = 1 - coord[:, :, 2]
        img_num = self.img_list[item].split('/')[2:]

        point_cloud = []
        shape_prior = []
        cate_list = []

        if self.source == 'CAMERA':
            cam_fx, cam_fy, cam_cx, cam_cy = self.camera_intrinsics
            cur_idx, cur_id, cur_instance = read_meta_file_camera(img_path)

            for i in range(len(cur_idx)):
                target = []
                if self.mode == 'train':
                    file_path = '{0}_bbox.txt'.format(
                        os.path.join(self.data_dir, 'gts', "train", img_num[0], img_num[1]),
                        cur_instance[i])
                    bbox = self.get_bbox_camera(file_path)
                    if int(cur_idx[i]) not in bbox:
                        continue
                    target = bbox[int(cur_idx[i])]
                    target = np.array(target)
                else:
                    file_path = '{0}/model_pts/{1}.txt'.format(os.path.join(self.data_dir, "others"), cur_instance[i])
                    if not os.path.exists(file_path):
                        continue
                    input_file = open(file_path, 'r')
                    for i in range(8):
                        input_line = input_file.readline()
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        input_line = input_line.split(' ')
                        target.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
                    input_file.close()
                    target = np.array(target)

                target_r, target_t = self.get_pose_camera(self.img_list[item].split('/')[2:], cur_idx[i])
                # target = self.enlarge_bbox(copy.deepcopy(target))
                target_tmp = np.dot(target, target_r.T) + target_t

                rmin, rmax, cmin, cmax = get_2dbbox(target_tmp, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale=1.0)

                idx = mask.copy()
                idx[:] = cur_idx[i]
                mask = np.equal(mask, idx)
                mask = np.logical_and(mask, depth > 0)

                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

                if len(choose) == 0:
                    continue
                if len(choose) > self.n_pts:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:self.n_pts] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, self.n_pts - len(choose)), 'wrap')
                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                pt2 = depth_masked / self.norm_scale
                pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
                cur_points = np.concatenate((pt0, pt1, pt2), axis=1)
                point_cloud.append(cur_points)

                # model = self.models[cur_instance[i]].astype(np.float32)  # 1024 points
                prior = self.mean_shapes[int(cur_id[i])].astype(np.float32)
                cate_list.append(cur_id[i])
                shape_prior.append(prior)
        else:
            cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics

            cur_idx, cur_id, cur_instance = read_meta_file(img_path)

            for i in range(len(cur_idx)):
                # mask = cv2.imread(img_path + '_mask.png')[:, :, 2]
                target = []
                input_file = open(
                    '{0}/model_scales/{1}.txt'.format(os.path.join(self.data_dir, "others"), cur_instance[i]), 'r')
                for j in range(8):
                    input_line = input_file.readline()
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                    input_line = input_line.split(' ')
                    target.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
                input_file.close()
                target = np.array(target)

                target_r, target_t = self.get_pose_real(self.img_list[item].split('/')[2:], cur_idx[i])

                target = self.enlarge_bbox(copy.deepcopy(target))
                target_tmp = np.dot(target, target_r.T) + target_t

                rmin, rmax, cmin, cmax = get_2dbbox(target_tmp, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale=1.0)

                idx = mask.copy()
                idx[:] = cur_idx[i]
                mask = np.equal(mask, idx)
                mask = np.logical_and(mask, depth > 0)

                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

                if len(choose) == 0:
                    continue
                if len(choose) > self.n_pts:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:self.n_pts] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, self.n_pts - len(choose)), 'wrap')
                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
                pt2 = depth_masked / self.norm_scale
                pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
                cur_points = np.concatenate((pt0, pt1, pt2), axis=1)
                point_cloud.append(cur_points)

                # model = self.models[cur_instance[i]].astype(np.float32)  # 1024 points
                prior = self.mean_shapes[int(cur_id[i])].astype(np.float32)
                cate_list.append(cur_id[i])
                shape_prior.append(prior)

        return point_cloud, shape_prior, cate_list

# aim = PoseDataset(source='CAMERA', mode='train', data_dir="/home/amax/document/sjt_project/datasets/MY_NOCS/",
#                   n_pts=1024)
# aim.__getitem__(6)
