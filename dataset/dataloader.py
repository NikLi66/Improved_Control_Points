import os
import pickle
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import random
import re
import cv2

from torch.utils import data


class DewarpDataSet(data.Dataset):
	def __init__(self, root, split='train', is_return_img_name=False):
		self.root = os.path.expanduser(root)
		self.split = split
		self.is_return_img_name = is_return_img_name
		self.images = collections.defaultdict(list)

		self.row_gap = 2
		self.col_gap = 2
		self.img_size = 992
		self.kernel_size = [1, 3, 5, 7, 9, 11, 13, 15]

		self.initial_dataset(split)

	def initial_dataset(self, split):
		self.images[split] = os.listdir(self.root)

	def __len__(self):
		return len(self.images[self.split])

	def __getitem__(self, item):
		if self.split == 'test':
			im_name = self.images[self.split][item]
			im_path = pjoin(self.root, im_name)

			img = cv2.imread(im_path, flags=cv2.IMREAD_COLOR)
			img = self.resize_im(img)
			img = self.transform_im(img)

			if self.is_return_img_name:
				return img, im_name
			return img

		name = self.images[self.split][item]
		with open(os.path.join(self.root, name), 'rb') as f:
			data = pickle.load(f)
		img, lbl, segment = data['image'], data['fiducial_points'], data['segment']

		box = self.get_bbox(lbl, img.shape[0], img.shape[1])
		img, M = self.crop(img, box)
		lbl = self.tran_points(lbl, M, box[0][0], box[0][1])

		img = self.resize_im(img)
		lbl = self.resize_lbl(lbl, img.shape[0], img.shape[1])
		lbl, segment = self.fiducal_points_lbl(lbl, segment)


		ker_size = random.choice(self.kernel_size)
		img = cv2.GaussianBlur(img, (ker_size, ker_size), 0)

		img = img.transpose(2, 0, 1)
		lbl = lbl.transpose(2, 0, 1)

		img = torch.from_numpy(img)
		lbl = torch.from_numpy(lbl).float()
		segment = torch.from_numpy(segment).float()

		if self.split == 'train':
			return img, lbl, segment
		elif self.split == 'val':
			return img, lbl, segment, name
		else:
			return img, name


	def crop(self, img, box):
		rect = cv2.minAreaRect(box)
		if rect[-1] < -45:
			rect = (rect[0], (rect[1][1], rect[1][0]), rect[-1] + 90)
		elif rect[-1] > 45:
			rect = (rect[0], (rect[1][1], rect[1][0]), rect[-1] - 90)
		w, h = int(rect[1][0]), int(rect[1][1])
		src, dst = np.array([box[3], box[0], box[1], box[2]], dtype="float32"), np.array([[0, h - 1],
																						  [0, 0],
																						  [w - 1, 0],
																						  [w - 1, h - 1]],
																						 dtype="float32")
		M = cv2.getPerspectiveTransform(src, dst)
		res = cv2.warpPerspective(img, M, (w, h), cv2.INTER_LINEAR, borderValue=(255, 255, 255))
		return res, M

	def tran_points(self, pts, M, xo, yo):
		for i in range(len(pts)):
			for j in range(len(pts[i])):
				pt = pts[i][j]
				pt = np.array([pt[0], pt[1], 1])
				new_pt = np.dot(M, pt)
				new_pt = new_pt / new_pt[2]
				pt[0] -= xo
				pt[1] -= yo
				pts[i][j] = new_pt[:2]
		return pts

	def get_bbox(self, fps, h, w):
		min_x, min_y, max_x, max_y = 1e9, 1e9, -1e9, -1e9
		for i in range(len(fps)):
			for j in range(len(fps[i])):
				fp = fps[i][j]
				min_x = min(min_x, fp[0])
				min_y = min(min_y, fp[1])
				max_x = max(max_x, fp[0])
				max_y = max(max_y, fp[1])
		min_x = random.randint(0, int(min_x))
		min_y = random.randint(0, int(min_y))
		max_x = random.randint(int(max_x), w)
		max_y = random.randint(int(max_y), h)

		return np.array([min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dtype='int32').reshape(-1, 2)

	def transform_im(self, img):
		img = img.transpose(2, 0, 1)
		img = torch.from_numpy(img).float()

		return img

	def resize_im(self, img):
		img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
		return img

	def resize_lbl(self, lbl, h ,w):
		lbl = lbl/[w, h]*[self.img_size, self.img_size]
		return lbl

	def fiducal_points_lbl(self, fiducial_points, segment):
		fiducial_points = fiducial_points[::self.row_gap, ::self.col_gap, :]
		segment = segment * [self.col_gap, self.row_gap]
		return fiducial_points, segment
