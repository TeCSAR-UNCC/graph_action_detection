import sys
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
import argparse
import cv2
from PIL import Image
import os.path as osp
import cv2
import math
from tqdm import tqdm
import time
import scipy.io as sci

from src.ReIDFeatureExtractor.tri_loss.model.Model import Model
from src.ReIDFeatureExtractor.tri_loss.utils.utils import load_state_dict
from src.ReIDFeatureExtractor.tri_loss.utils.utils import set_devices
from src.ReIDFeatureExtractor.tri_loss.utils.dataset_utils import get_im_names
from src.ReIDFeatureExtractor.tri_loss.utils.distance import normalize

def preProcessReIDCrop(image):
    image = cv2.resize(image, (128,256), interpolation=cv2.INTER_LINEAR)
    image = image / 255
    image = image - np.array([0.486, 0.459, 0.408])
    image = image / np.array([0.229, 0.224, 0.225]).astype(float)
    image = image.transpose(2,0,1)
    return image

def getReIDCrops(image, bboxes, valid_keypoint_perc, keypoint_threshold):
    imheight, imwidth = image.shape[:2]

    crops = []
    bad_boxes = []
    for idx in range(bboxes.shape[0]):
        bbox = bboxes[idx,:]
        keypoint_count = valid_keypoint_perc[idx]
        if (keypoint_count >= keypoint_threshold):
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[0] + bbox[2])
            y2 = int(bbox[1] + bbox[3])

            x1 = min(imwidth, max(x1,0))
            y1 = min(imheight, max(y1,0))
            x2 = min(imwidth, max(x2,0))
            y2 = min(imheight, max(y2,0))

            area = (x2-x1)*(y2-y1)
            if area > 0:
                crop = image[y1:y2, x1:x2]
                crop = preProcessReIDCrop(crop)
                crops.append(crop)
                bad_boxes.append(False)
            else:
                crop = np.zeros((256,128,3))
                crop = preProcessReIDCrop(crop)
                crops.append(crop)
                bad_boxes.append(True)
        else:
            crop = np.zeros((256,128,3))
            crop = preProcessReIDCrop(crop)
            crops.append(crop)
            bad_boxes.append(True)
    crops = np.array(crops, dtype=float)
    return crops, bad_boxes

class FeatureEncoder():
    def __init__(self, checkpoint_path, using_gpu):
        self.checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.net = Model(net='mobilenetV2', pretrained=False)
        self.gpu = using_gpu
    def encodeFeatures(self, image, bboxes, keypoint_count, keypoint_threshold):
        load_state_dict(self.net, self.checkpoint)
        
        self.net = self.net.eval()
        if self.gpu:
            self.net = self.net.cuda()

        reidcrops, bad_dets = getReIDCrops(image, bboxes, keypoint_count, keypoint_threshold)
        imgs = torch.from_numpy(reidcrops).float()
        if self.gpu:
            imgs = imgs.cuda()
        
        with torch.no_grad():
            feats = self.net(imgs)
        feats = feats.data.cpu().numpy()
        for j in range(len(bad_dets)):
            if bad_dets[j] == True:
                feats[j,:] = np.zeros(1280)

        return feats