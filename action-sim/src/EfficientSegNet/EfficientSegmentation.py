import os

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data
import torch.utils.data.distributed
import torchvision.transforms
from src.EfficientHRNetKeypoints.utils.transforms import resize

import torch.backends.cudnn as cudnn

from src.EfficientSegNet.config import seg_cfg
from src.EfficientSegNet.EfficientNetSeg import get_seg_model

class EfficientSegNet():
    def __init__(self, seg_cfg):
        self.seg_cfg = seg_cfg
        self.net = get_seg_model(seg_cfg, is_train=False)
        self.crop_size = (1024,1024)
        self.base_size = 1024
        self.num_classes = seg_cfg.DATASET.NUM_CLASSES
        
    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=padvalue)
        
        return pad_image
   
    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                                (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                                (self.ignore_label,))
        
        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label


    def multi_scale_aug(self, image, label=None, 
            rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)
        
        image = cv2.resize(image, (new_w, new_h), 
                           interpolation = cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), 
                           interpolation = cv2.INTER_NEAREST)
        else:
            return image
        
        if rand_crop:
            image, label = self.rand_crop(image, label)
        
        return image, label

    def inference(self, model, image, flip=False):
        size = image.size()
        pred = model(image)
        pred = F.upsample(input=pred, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear')        
        if flip:
            flip_img = image.numpy()[:,:,:,::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear')
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy())#.cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()
    
    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        #image.cpu()
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width])#.cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                #print("crop", new_img.shape)
                        
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w])#.cuda()
                count = torch.zeros([1,1, new_h, new_w])#.cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred

    
    def runSegmentation(self,image):
        cudnn.benchmark = seg_cfg.CUDNN.BENCHMARK
        cudnn.deterministic = seg_cfg.CUDNN.DETERMINISTIC
        cudnn.enabled = seg_cfg.CUDNN.ENABLED

        model_state_file = seg_cfg.TEST.MODEL_FILE
        #print(model_state_file)
        pretrained_dict = torch.load(model_state_file)
        model_dict = self.net.state_dict()
        #print(model_dict)
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        #print(pretrained_dict)
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)

        #self.net.load_state_dict(torch.load(seg_cfg.TEST.MODEL_FILE, map_location='cpu'), strict=True)
        self.net.eval()
        #print("here")
        
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
    
            ]
        )
        image = transforms(image)
        image = image.unsqueeze(0)#.cuda()
        final_pred = self.multi_scale_inference(self.net,image)
        #print("Final pred", final_pred.size())
        np_final_pred = final_pred.permute(0,1,2,3).cpu().detach().numpy()
        return np_final_pred
