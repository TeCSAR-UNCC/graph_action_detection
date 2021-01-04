from src import Options
from src.PersonIDType import PersonIDType
from csv import reader
import os
import cv2
import numpy as np
import imutils
from numpy import linalg as la # L2Norm calculations
import json # Used for OpenPose keypoint json files
import statistics
from sklearn.ensemble import IsolationForest
import sys
from PIL import Image
import math
import csv
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import torch
import torch.nn as nn

use_EfficientHRNet = True
use_EfficientSegNet = False

eval_set_2_valid_frames = [4076,12456]

bbox_colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[177,177,0],[177,0,177],[0,177,177]]
grey_box = [177,177,177]





def upscale_xywh_bbox(bbox, width, height):
    x = bbox[0] * width
    y = bbox[1] * height
    w = bbox[2] * width
    h = bbox[3] * height

    return np.array([x,y,w,h], dtype=int)

def xywh2xyxy(bbox):
    bbox = bbox.reshape(4)
    x0 = bbox[0]
    x1 = bbox[0] + bbox[2]
    y0 = bbox[1]
    y1 = bbox[1] + bbox[3]

    newbbox = [x0, y0, x1, y1]
    return np.array(newbbox, dtype=int)

def xyxy2xywh(bbox):
    bbox = bbox.reshape(4)
    x = bbox[0]
    y = bbox[1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    return np.array([x,y,w,h], dtype=int)

def IoU(bbox0, bbox1):
    _bbox0 = xywh2xyxy(bbox0)
    x0_1 = _bbox0[0]
    x0_2 = _bbox0[2]
    y0_1 = _bbox0[1]
    y0_2 = _bbox0[3]

    _bbox1 = xywh2xyxy(bbox1)
    x1_1 = _bbox1[0]
    x1_2 = _bbox1[2]
    y1_1 = _bbox1[1]
    y1_2 = _bbox1[3]

    if ((x0_1 > x1_2) or (x0_2 < x1_1) or (y0_1 > y1_2) or (y0_2 < y1_1)):
        return 0
    else:
        dx = min((x0_2, x1_2)) - max((x0_1, x1_1))
        dy = min((y0_2, y1_2)) - max((y0_1, y1_1))
        area = bbox0[2] * bbox0[3] + bbox1[2] * bbox1[3]
        intersection = dx * dy
        union = area - intersection
        return intersection/union

class ObjectHistory():
    def __init__(self, life, sendObject, keyCount=0, reIDFlag=0, sentToServer=0):
        self.life = life
        self.sendObject = sendObject
        self.keyCount = keyCount
        self.keypoints = np.zeros((1))
        self.reIDFlag = reIDFlag
        self.sentToServer = sentToServer
        self.seen = 0
        
        

class actionbuffer():
    def __init__(self):
        self.KBufferTotal=0
        self.SpaceTimeGraph= torch.rand(1,2,30,18,1)
        self.rank=0
        self.actionBool=0

class EdgeNode():
    def __init__(self, cam_id, opts, frontend_net, Keyprocessor ,seg_frontend_net, id_encoder, max_feats=1, gt=[]):
        self.id = cam_id
        self.opts = opts
        self.iou_weight = 0.3
        self.l2_weight = 0.7
        self.local_match_threshold = 2
        self.keypoint_threshold = 0.55
        self.server_keypoint_threshold = 0.8
        self.server_seen_threshold = 10
        self.table_size = 200
        
        self.Keyprocessor= Keyprocessor
        self.actionbuffer=actionbuffer()
        self.IDDict=dict()
        self.classlabel=dict()
        self.classlabel['0']='Throwing'
        self.classlabel['1']='Sitting down'
        self.classlabel['2']='Jumping'
        self.classlabel['3']='Staggering'
        self.classlabel['4']='Walking'
        self.currFrameNum=0
        self.fileGTBool=0
        self.ansGT=[]
        self.hipGT=[]
        self.TryList=[]
        self.HIP_TryList=[]
        self.errorlist=[]
        self.LastHipInfrence=0
        
        
        
        
        table_life_in_seconds = 10
        self.table_life = max(int(table_life_in_seconds * min(opts.framerate, opts.source_framerate)),1)
        self.id_table = [ObjectHistory(-1, PersonIDType()) for i in range(self.table_size)]
        self.sendQ = [] # Holds PersonIDType list for edge server to identify
        self.recvQ = [] # Holds [oldID, newID] list from server for updating IDs in id_table
        self.currLabel = cam_id * 1000000
        if (len(gt) > 0):
            self.gt = gt.reshape((gt.shape[1], gt.shape[2]))
            self.eval = True
            self.validframes = np.sort(np.unique(np.transpose(self.gt[:,2])))
        elif (opts.eval_set >= 2):
            self.gt = gt
            self.eval = False
            startFrame = eval_set_2_valid_frames[0]
            endFrame = eval_set_2_valid_frames[1]
            self.validframes = [i for i in range(startFrame,endFrame)]
        else:
            self.gt = gt
            self.eval = False
            self.validframes = [0]
        self.gt_match_threshold = 0.3
        self.frontend_net = frontend_net
        self.seg_frontend_net = seg_frontend_net
        self.id_encoder = id_encoder
        self.using_openpose = ((opts.eval_set % 2) == 1)
        self.max_feats = max_feats
        
        #create the filter
        self.my_filter = KalmanFilter(dim_x=2, dim_z=1)
        self.my_filter.x = np.array([[0],[0.1]])# initial state (location and velocity)
        self.my_filter.F = np.array([[1.,1.],[0.,1.]])    # state transition matrix (dim_x, dim_x)
        self.my_filter.H = np.array([[1.,0.]])    # Measurement function  (dim_z, dim_x)
        self.my_filter.P *= 1000.                 # covariance matrix ( dim x, dim x) 
        self.my_filter.R = 5                      # state uncertainty,#measurement noise vector  (dim_z, dim_z)
        self.my_filter.Q = Q_discrete_white_noise(dim=2, dt=2, var=0.5)# process uncertainty  (dim_x, dim_x)
        
        
    def kfilter(self,angle):
        self.my_filter.predict()
        self.my_filter.update(angle)
        return self.my_filter.x

    def getMobileNetPoseData(self, frame, image):
        # Calculate poses and regress bounding boxes from keypoints
        bboxlist = []
        keypoints = []
        valid_keypoint_perc = []
        poses = self.frontend_net.runPoseEstimation(image)
        for pose in poses:
            # bbox has the form [x1, y1, w, h]
            valid_keypoint_perc.append(pose.num_found_keypoints/len(pose.keypoints))
            bboxlist.append(pose.bbox)
            keypoints.append(pose.keypoints)
        bboxes = np.array(bboxlist)
        keypoints = np.array(keypoints)
        valid_keypoint_perc = np.array(valid_keypoint_perc)

        return bboxes, keypoints, valid_keypoint_perc

    def getEfficientSegNetData(self, frame,image):
        segmented_image = self.seg_frontend_net.runSegmentation(image)

        preds = np.asarray(np.argmax(segmented_image, axis=1), dtype=np.uint8)

        mask = np.zeros((720,1280,3), dtype=np.int8)

        for i in range(len(preds[0])):
            for j in range(len(preds[0][0])):
                temp = preds[0][i][j]
                if temp == 0: #road
                    mask[i][j][0] = 69
                    mask[i][j][1] = 78
                    mask[i][j][2] = 206
                elif temp == 1: #sidewalk
                    mask[i][j][0] = 102
                    mask[i][j][1] = 111
                    mask[i][j][2] = 247
                elif temp == 2: #building
                    mask[i][j][0] = 93
                    mask[i][j][1] = 223
                    mask[i][j][2] = 128
                elif temp == 3: #fence
                    mask[i][j][0] = 128
                    mask[i][j][1] = 223
                    mask[i][j][2] = 93
                elif temp == 4: #wall
                    mask[i][j][0] = 131
                    mask[i][j][1] = 240
                    mask[i][j][2] = 145
                elif temp == 5: #veg
                    mask[i][j][0] = 153
                    mask[i][j][1] = 255
                    mask[i][j][2] = 153
                elif temp == 6: #terrain
                    mask[i][j][0] = 153
                    mask[i][j][1] = 255
                    mask[i][j][2] = 153
                elif temp == 7: #car
                    mask[i][j][0] = 57
                    mask[i][j][1] = 135
                    mask[i][j][2] = 223
                elif temp == 8: #cycle
                    mask[i][j][0] = 57
                    mask[i][j][1] = 135
                    mask[i][j][2] = 223
                elif temp == 9: #bus
                    mask[i][j][0] = 57
                    mask[i][j][1] = 135
                    mask[i][j][2] = 223
                elif temp == 10: #train
                    mask[i][j][0] = 57
                    mask[i][j][1] = 135
                    mask[i][j][2] = 223
                elif temp == 11: #truck
                    mask[i][j][0] = 228
                    mask[i][j][1] = 70
                    mask[i][j][2] = 38
                elif temp == 12: #motorcycle
                    mask[i][j][0] = 57
                    mask[i][j][1] = 135
                    mask[i][j][2] = 223
                elif temp == 13: #sky
                    mask[i][j][0] = 204
                    mask[i][j][1] = 102
                    mask[i][j][2] = 204
                elif temp == 14: #pole
                    mask[i][j][0] = 204
                    mask[i][j][1] = 102
                    mask[i][j][2] = 204
                elif temp == 15: #traffic sign
                    mask[i][j][0] = 255
                    mask[i][j][1] = 153
                    mask[i][j][2] = 255
                elif temp == 16: #traffic light
                    mask[i][j][0] = 255
                    mask[i][j][1] = 153
                    mask[i][j][2] = 255
                elif temp == 17: #person
                    mask[i][j][0] = 228
                    mask[i][j][1] = 70
                    mask[i][j][2] = 38
                elif temp == 18: #rider
                    mask[i][j][0] = 228
                    mask[i][j][1] = 70
                    mask[i][j][2] = 38
                else:
                    mask[i][j][0] = 0
                    mask[i][j][1] = 0
                    mask[i][j][2] = 0

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        output = ((0.0 * image) + (1.0 * mask)).astype("uint8")
        return output

    def getEfficientHRNetPoseData(self, frame, image, outlier_thresh=-0.1):
        # Calculate poses and regress bounding boxes from keypoints
        valid_keypoint_perc = []
        heatmap_th = 0.075
        keypoints, x_coordinates, y_coordinates, h_scores = self.frontend_net.runPoseEstimation(image)
        #print("Keypoints length", len(keypoints))
        bboxes = []
        final_keypoints = []
        for idx in range(len(keypoints)):
            # Remove Outliers
            temp_xy = []
            for i in range(len(keypoints[idx])):
                temp_x = x_coordinates[idx][i]
                temp_y = y_coordinates[idx][i]
                temp_xy.append([temp_x, temp_y])
            clf = IsolationForest(random_state=0,behaviour='deprecated').fit(temp_xy)
            clf_scores = clf.decision_function(temp_xy)
            #print("CLF scores",clf_scores)
            for clf_idx in range(len(clf_scores)):
                if clf_scores[clf_idx] < outlier_thresh:
                    keypoints[idx][clf_idx][0] = -1
                    keypoints[idx][clf_idx][1] = -1
            ############################
            num_found_keypoints = 0
            x_ax = []
            y_ax = []
            for i in range(len(keypoints[idx])):
                #print("index i", i)
                if (keypoints[idx][i][0] and keypoints[idx][i][1]) != -1:
                    if h_scores[idx][i] >= heatmap_th:
                        num_found_keypoints += 1
                        x_coor = x_coordinates[idx][i]
                        x_ax.append(x_coor)
                        y_coor = y_coordinates[idx][i]
                        y_ax.append(y_coor)
            if len(x_ax) > 0 and len(y_ax) > 0 :
                x_min = np.amin(x_ax)
                y_min = np.amin(y_ax)
                width = np.amax(x_ax) - x_min
                height = np.amax(y_ax) - y_min
                if(height >= width):
                    bboxes.append([x_min, y_min, width, height])
                    valid_keypoint_perc.append(num_found_keypoints/len(keypoints[idx]))
                    final_keypoints.append(keypoints[idx])

        valid_keypoint_perc = np.array(valid_keypoint_perc)
        bboxes = np.array(bboxes)
        keypoints = np.array(final_keypoints)
        return bboxes, keypoints, valid_keypoint_perc

    def getOpenPoseData(self, frame, image):
        pose_path = self.opts.json_path + '/camera' + str(self.id) + "/{:06d}_keypoints.json".format(int(frame))
        im_h = image.shape[0]
        im_w = image.shape[1]
        with open(pose_path) as pose_file:
            pose_data = json.load(pose_file)
        pose_people = pose_data['people']
        bboxes = []
        keypoints = []
        valid_keypoint_perc = []
        for person in pose_people:
            person_keypoints = person['pose_keypoints_2d']
            if len(person_keypoints) > 0:
                person_keypoints = np.array(person_keypoints)
                person_keypoints = person_keypoints.reshape((25,3))
                # person_keypoints = person_keypoints[:18,:] # Ignore foot keypoints in BODY_25
                person_keypoints[:,0] = person_keypoints[:,0] * im_w
                person_keypoints[:,1] = person_keypoints[:,1] * im_h
                confidence_threshold = 0.05
                found_keypoints = np.zeros((np.count_nonzero(person_keypoints[:, 2] > confidence_threshold), 2), dtype=np.int32)
                found_kpt_id = 0
                for kpt_id in range(person_keypoints.shape[0]):
                    if person_keypoints[kpt_id, 2] <= confidence_threshold:
                        continue
                    found_keypoints[found_kpt_id,0] = person_keypoints[kpt_id,0]
                    found_keypoints[found_kpt_id,1] = person_keypoints[kpt_id,1]
                    found_kpt_id += 1
                bbox = cv2.boundingRect(found_keypoints)

                bboxes.append(bbox)
                keypoints.append(person_keypoints)
                valid_keypoint_perc.append(found_keypoints.shape[0] / person_keypoints.shape[0])

        bboxes = np.array(bboxes)
        keypoints = np.array(keypoints)
        valid_keypoint_perc = np.array(valid_keypoint_perc)
        
        return bboxes, keypoints, valid_keypoint_perc

    def getFrameFeatures(self, frame):
        image_path = self.opts.image_path + '/camera' + str(self.id) + '/' + "{:06d}".format(int(frame)) + '.jpg'
        bgr_image = cv2.imread(image_path)
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        #print(image)
        #print(image.shape)

        if self.using_openpose:
            bboxes, keypoints, valid_keypoint_perc = self.getOpenPoseData(frame, image)
        elif use_EfficientHRNet:
            bboxes, keypoints, valid_keypoint_perc = self.getEfficientHRNetPoseData(frame, image)
        else:
            bboxes, keypoints, valid_keypoint_perc = self.getMobileNetPoseData(frame, image)

        encodedFeatures = []
        # Crop images and extract encoded features
        if bboxes.size > 0:
            encodedFeatures = self.id_encoder.encodeFeatures(image, bboxes, valid_keypoint_perc, self.keypoint_threshold)
        #print(image.shape)
        if use_EfficientSegNet:
            image = self.getEfficientSegNetData(frame,image)
        
        return image, bboxes, keypoints, valid_keypoint_perc, encodedFeatures

    def drawFrameBoxes(self, frame, image, ObjectHistoryList):
        image_path = self.opts.image_output_path + '/camera' + str(self.id)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        image_path += '/' + "{:06d}".format(int(frame)) + '.jpg'

        for objhist in ObjectHistoryList:
            personID = objhist.sendObject
            
            if (objhist.reIDFlag == 1):
                coloridx = int(personID.label % 9)
                colorlist = bbox_colors[coloridx]
            else:
                colorlist = grey_box
            
            bbox = xywh2xyxy(personID.bbox)
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            color = (colorlist[0], colorlist[1], colorlist[2])
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            thickness = 2
            id_label = "{:0d}".format(int(personID.label % 1000000))
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            image = cv2.putText(image, id_label, start_point, font, scale, color, thickness)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image)

    def drawPoseBody25(self, image, objhist):
        kp_pairs = [[0,1],    # Neck
                    [0,15],   # Nose -> R Eye
                    [0,16],   # Nose -> L Eye
                    [15,17],  # R Eye -> R Ear
                    [16,18],  # L Eye -> L Ear
                    [1,2],    # R Collarbone
                    [1,5],    # L Collarbone
                    [1,8],    # Torso
                    [2,3],    # R Bicep
                    [3,4],    # R Forearm
                    [5,6],    # L Bicep
                    [6,7],    # L Forearm
                    [8,9],    # R Waist
                    [8,12],   # L Waist
                    [9,10],   # R Thigh
                    [10,11],  # R Calf
                    [12,13],  # L Thigh
                    [13,14],  # L Calf
                    # [11,22],  # R Foot
                    # [11,24],  # R Heel
                    # [22,23],  # R Toes
                    # [14,19],  # L Foot
                    # [14,21],  # L Heel
                    # [19,20]   # L Toes
        ]

        keypoints = objhist.keypoints
        personID = objhist.sendObject

        if (objhist.reIDFlag == 1):
            coloridx = int(personID.label % 9)
            colorlist = bbox_colors[coloridx]
        else:
            colorlist = grey_box

        bbox = xywh2xyxy(personID.bbox)
        start_point = (bbox[0], bbox[1])
        color = (colorlist[0], colorlist[1], colorlist[2])
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        start_point2 = (bbox[0], bbox[1]-25)
        if actionbool == 1 :
            image = cv2.putText(image, self.classlabel[str(rank)], start_point2, font, scale*1.2, color, thickness)
            #print(self.classlabel[str(rank)])
        # id_label = str(int(personID.label / 1000000)) + "-{:06d}".format(int(personID.label % 1000000))
        #id_label = "P-" + str(int(personID.label % 1000000))
        #image = cv2.putText(image, id_label, start_point, font, scale, color, thickness)

        for pair in kp_pairs:
            kp0 = keypoints[pair[0],:]
            kp1 = keypoints[pair[1],:]
            if (kp0[2] > 0.05) and (kp1[2] > 0.05):
                image = cv2.line(image, (int(kp0[0]), int(kp0[1])), (int(kp1[0]), int(kp1[1])), color, thickness)
            if (kp0[2] > 0.05):
                image = cv2.circle(image, (int(kp0[0]), int(kp0[1])), 3, color)
            if (kp1[2] > 0.05):
                image = cv2.circle(image, (int(kp1[0]), int(kp1[1])), 3, color)

        return image

    def drawPoseCOCO18(self, image, objhist, rank,actionbool,frame,numofOBJ):
    
    
        ansfile="/home/justin/Documents/mobility/other/ans.csv"
        HIPansfile="/home/justin/Documents/mobility/other/hip-ans.csv"
        hipGTFile="/home/justin/Documents/mobility/other/AVG-HIP-1.csv"
        if self.fileGTBool==0:
            
            GTfile="/home/justin/Documents/mobility/other/AVGknee1.csv"
            errorfile="/home/justin/Documents/mobility/other/error.csv"
            with open(GTfile, 'r') as read_obj:
                csv_reader = reader(read_obj)
                for row in csv_reader:
                    self.ansGT.append(float(row[0]))
                    
                    
            with open(hipGTFile, 'r') as read_obj:
                csv_reader = reader(read_obj)
                for row in csv_reader:
                    self.hipGT.append(float(row[0]))
            self.fileGTBool=1

        kp_pairs = [[14,16],
                    [13,15],
                    [12,14],
                    [11,13],
                    [6,8],
                    [8,10],
                    [5,7],
                    [7,9],
                    [1,3],
                    [2,4],
                    [3,5],
                    [4,6],
                    [0,1],
                    [0,2],
                    [5,11],
                    [6,12],
                    [5,6],
                    [11,12]]
                    
#11->13->15
#12->14->16
                    

        keypoints = objhist.keypoints
        personID = objhist.sendObject

        if (objhist.reIDFlag == 1):
            coloridx = int(personID.label % 9)
            colorlist = bbox_colors[coloridx]
        else:
            colorlist = grey_box

        bbox = xywh2xyxy(personID.bbox)
        start_point = (bbox[0], bbox[1]-2)
        #color = (colorlist[0], colorlist[1], colorlist[2])
        color = (255, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        #id_label = str(int(personID.label / 1000000)) + "-{:01d}".format(int(personID.label % 1000000))
        #id_label = '{:d}'.format(int(personID.label / 1000000))
        #image = cv2.putText(image, id_label, start_point, font, scale, color, thickness)
        #action
        start_point2 = (bbox[0], bbox[1]-25)
        #if actionbool == 1 :
            #image = cv2.putText(image, self.classlabel[str(rank)], start_point2, font, scale*1.2, color, thickness)
            #print(self.classlabel[str(rank)])

        for pair in kp_pairs:
        #print(keypoints.shape)
            kp0 = keypoints[pair[0],:]
            kp0Name=pair[0]
            kp1 = keypoints[pair[1],:]
            kp1Name=pair[1]

            if (kp0[0] != -1) and (kp1[0] != -1):
                image = cv2.line(image, (int(kp0[0]), int(kp0[1])), (int(kp1[0]), int(kp1[1])), color, 6)
            if (kp0[0] != -1):
                image = cv2.circle(image, (int(kp0[0]), int(kp0[1])), 3, color)
                
                #image = cv2.putText(image, str(kp0Name), (int(kp0[0]), int(kp0[1])), font, scale*1.2, color, thickness)
                
            if (kp1[0] != -1):
                image = cv2.circle(image, (int(kp1[0]), int(kp1[1])), 3, color)
                #image = cv2.putText(image, str(kp1Name), (int(kp1[0]), int(kp1[1])), font, scale*1.2, color, thickness)
            #if (kp0Name == 11) and (kp1Name == 13) and (kp0[0] != -1) and (kp1[0] != -1):
            #   point1=(int(kp0[0]), int(kp0[1]))
            #   point2=(int(kp1[0]), int(kp1[1]))
            #elif (kp0Name == 13) and (kp1Name == 15)
            #   point3=(int(kp1[0]), int(kp1[1]))
            
        sho_L = keypoints[5,:]
        sho_R = keypoints[6,:]
        
        kp0L = keypoints[11,:]
        kp1L = keypoints[13,:]
        kp2L = keypoints[15,:]
        kp0R = keypoints[12,:]
        kp1R = keypoints[14,:]
        kp2R = keypoints[16,:]
        #self.TryList=[]
        #self.errorlist=[]
        answer=0

        if (kp0L[0] != -1) and (kp1L[0] != -1) and (kp2L[0] != -1) and (numofOBJ ==0):
            A= math.sqrt( ((kp0L[0]-kp1L[0])**2)+ ((kp0L[1]-kp1L[1])**2) )
            B= math.sqrt( ((kp1L[0]-kp2L[0])**2)+ ((kp1L[1]-kp2L[1])**2) )
            C= math.sqrt( ((kp0L[0]-kp2L[0])**2)+ ((kp0L[1]-kp2L[1])**2) )
            answer=180-math.degrees(math.acos( ( (C**2)-(A**2)-(B**2) )/(-2*A*B) ) )
            self.TryList.append(answer)

            #print("the answer is :",answer,"\n")
        elif (kp0R[0] != -1) and (kp1R[0] != -1) and (kp2R[0] != -1) and (numofOBJ ==0):
            A= math.sqrt( ((kp0R[0]-kp1R[0])**2)+ ((kp0R[1]-kp1R[1])**2) )
            B= math.sqrt( ((kp1R[0]-kp2R[0])**2)+ ((kp1R[1]-kp2R[1])**2) )
            C= math.sqrt( ((kp0R[0]-kp2R[0])**2)+ ((kp0R[1]-kp2R[1])**2) )
            answer=180-math.degrees(math.acos( ( (C**2)-(A**2)-(B**2) )/(-2*A*B) ) )

            self.TryList.append(answer)
            
        elif  (numofOBJ ==0):
            #print("the answer is error")
            answer=-1
            self.TryList.append(answer)
        if (answer!= -1) and (numofOBJ ==0):
           error= (abs(self.ansGT[int(frame-1)]-answer)/self.ansGT[int(frame-1)])*100
           ep = (1270, 370)
           self.errorlist.append(error)
           report=str(answer)
           report= report[0:6]
           Fronttext="Knee bend : "
           errorPres= Fronttext +report +" degs"
           image = cv2.putText(image, errorPres,ep, font, scale*1.2, (255, 0, 0), thickness)
          
           ep = (1270, 410)
           KneeGt=str(self.ansGT[int(frame-1)])
           KneeGt=KneeGt[0:6]
           Fronttext="Knee bend GT : "
           GTPres= Fronttext +KneeGt +" degs"
           image = cv2.putText(image, GTPres,ep, font, scale*1.2, (0, 0, 255), thickness)
           

           if int(frame)==175:
              print(self.errorlist)
              print(self.TryList)
              

        if (kp0L[0] != -1) and (kp1L[0] != -1) and (sho_L[0] != -1) and (numofOBJ ==0):

            A= math.sqrt( ((sho_L[0]-kp0L[0])**2)+ ((sho_L[1]-kp0L[1])**2) )
            B= math.sqrt( ((kp0L[0]-kp1L[0])**2)+ ((kp0L[1]-kp1L[1])**2) )
            C= math.sqrt( ((sho_L[0]-kp1L[0])**2)+ ((sho_L[1]-kp1L[1])**2) )
            try:
                HIPanswer=180-math.degrees(math.acos( ( (C**2)-(A**2)-(B**2) )/(-2*A*B) ) )
                self.LastHipInfrence=HIPanswer
            except:
                HIPanswer =float(self.kfilter( self.LastHipInfrence)[0])
                



                
                
                            
            self.HIP_TryList.append(HIPanswer)
            #print("the answer is :",answer,"\n")
        elif (kp0R[0] != -1) and (kp1R[0] != -1) and (sho_R[0] != -1) and (numofOBJ ==0):
            
            #hip
            A= math.sqrt( ((sho_R[0]-kp0R[0])**2)+ ((sho_R[1]-kp0R[1])**2) )
            B= math.sqrt( ((kp0R[0]-kp1R[0])**2)+ ((kp0L[1]-kp1R[1])**2) )
            C= math.sqrt( ((sho_R[0]-kp1R[0])**2)+ ((sho_R[1]-kp1R[1])**2) )
            try:
                HIPanswer=180-math.degrees(math.acos( ( (C**2)-(A**2)-(B**2) )/(-2*A*B) ) )
                self.LastHipInfrence=HIPanswer
            except:
               HIPanswer =float(self.kfilter( self.LastHipInfrence)[0])
            self.HIP_TryList.append(HIPanswer)
            #print("the answer is :",answer,"\n")
        elif  (numofOBJ ==0):
            #print("the answer is error")
            HIPanswer =float(self.kfilter( self.LastHipInfrence)[0])
            self.HIP_TryList.append(HIPanswer)
        if (answer!= self.LastHipInfrence) and (numofOBJ ==0):
           
           #hip           
           error= (abs(self.hipGT[int(frame-1)]-HIPanswer)/self.hipGT[int(frame-1)])*100
           ep = (1270, 290)
           report=str(HIPanswer)
           report= report[0:4]
           Fronttext="Hip bend : "
           errorPres= Fronttext +report +" degs"
           image = cv2.putText(image, errorPres,ep, font, scale*1.2, (255, 0, 0), thickness)
           
           
           ep = (1270, 330)
           HIPGtString=str(self.hipGT[int(frame-1)])
           HIPGtString=HIPGtString[0:6]
           Fronttext="Hip bend GT : "
           GTPres= Fronttext +HIPGtString +" degs"
           image = cv2.putText(image, GTPres,ep, font, scale*1.2, (0, 0, 255), thickness)
           
              
              
           with open(ansfile, mode='w') as Pred_angle:
              Pred_angle_writer = csv.writer(Pred_angle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
              for i in range(len(self.TryList)):
                 Pred_angle_writer.writerow([self.TryList[i]])

           with open(HIPansfile, mode='w') as Pred_angle:
              Pred_angle_writer = csv.writer(Pred_angle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
              for i in range(len(self.HIP_TryList)):
                 Pred_angle_writer.writerow([self.HIP_TryList[i]])

                
                
     

#11->13->15
#12->14->16
        return image

    def drawFramePoses(self, frame, image, ObjectHistoryList):
        image_path = self.opts.image_output_path + '/camera' + str(self.id)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        image_path += '/' + "{:06d}".format(int(frame)) + '.jpg'
        numofOBJ=0
        for objhist in ObjectHistoryList:
            if (self.opts.eval_set % 2 == 0):
                keypoints = objhist.keypoints   
                personID = objhist.sendObject
                IDkey=personID.label
                if len(self.IDDict) == 0  or IDkey not in self.IDDict:
                   self.IDDict[IDkey]=actionbuffer()
                   #print('\nnew buffer!\n')
                #problem this shares memory
                badKPtensor = torch.from_numpy(keypoints[:,:].copy())#Keypoint tensor
                #print(type(KPtensor))
                KPtensor=torch.rand(18,2)
                old_kp =[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
                NEW_kp =[0,5,7,9,6,8,10,11,13,15,12,14,16,1,2,3,4]
                for i in range(len(NEW_kp)):
             	    for c in range(2): #channels
              	        x=NEW_kp[i]
              	        y=old_kp[i]
              	        KPtensor[y,c]=badKPtensor[x,c]
                KPtensor[1,0]=(KPtensor[5,0]+KPtensor[6,0]/2)
                KPtensor[1,1]=(KPtensor[5,1]+KPtensor[6,1]/2)
	    
                #to do set keypoint 1 = to new keypoint 5 and 6 averaged
                #im going from their new to my old
               # print(KPtensor.size())
                xCenter=((KPtensor[0,0]+ KPtensor[8,0]+KPtensor[11,0])/3)
                yCenter=((KPtensor[0,1]+ KPtensor[8,1]+KPtensor[11,1])/3)
                for i in range(17):
                    KPtensor[i,0]=KPtensor[i,0]-xCenter
                    KPtensor[i,1]=KPtensor[i,1]-yCenter
                #print(KPtensor)
                #Input shape should be (N, C, T, V, M)
                #print('\nexecute\n')
                KPtensor = KPtensor.permute(1,0)
                KPtensor.unsqueeze_(-1)
                KPtensor.unsqueeze_(-1)
                KPtensor.unsqueeze_(-1)
                #original shape=[channels, joints, time, samples, people])
                KPtensor = KPtensor.permute(3,0,2,1,4)
                #new shape: [samples,channels,time,joints,people]
                #print(KPtensor.size())#=print(KPtensor.shape)
                #print(KPtensor)

                #print('We are at frame :',self.KBufferTotal)

                if self.IDDict[IDkey].KBufferTotal == 29:
                   self.IDDict[IDkey].SpaceTimeGraph=torch.roll(self.IDDict[IDkey].SpaceTimeGraph, -1, 0)
                   temp=self.IDDict[IDkey].SpaceTimeGraph[:,:,:self.IDDict[IDkey].KBufferTotal-1,:,:]
                   #        self.SpaceTimeGraph=torch.rand(1,2,150,18,1)
                   #print('temp shape :',temp.shape)
                   #print('KPtensor shape :',KPtensor.shape)
                   self.IDDict[IDkey].SpaceTimeGraph=torch.cat([temp,KPtensor.float()], 2)
                   output=self.Keyprocessor.start(self.IDDict[IDkey].SpaceTimeGraph.cuda())
                   rank = output.argsort()
                  # print(rank)
                   self.IDDict[IDkey].actionBool=1
                   self.IDDict[IDkey].rank=rank[0][4]#argsort goes through the reverse order, so the last is biggest
                  # print(rank,'rank \n')
                  # print(rank[0])
                   #print(output)
                   #new shape: [no,no,yes,no,no]

                   #print(self.SpaceTimeGraph)
                else:
                   self.IDDict[IDkey].KBufferTotal=self.IDDict[IDkey].KBufferTotal+1
                   temp=self.IDDict[IDkey].SpaceTimeGraph[:,:,:(self.IDDict[IDkey].KBufferTotal-1),:,:]
                   #print('KBuffershape :',self.IDDict[IDkey].SpaceTimeGraph.shape)
                   #print('temp shape :',temp.shape)
                   #print('KPtensor shape :',KPtensor.shape)
                   #        self.SpaceTimeGraph=torch.rand(1,2,150,18,1)
                   self.IDDict[IDkey].SpaceTimeGraph=torch.cat([temp,KPtensor.float()],2)
                   #print(self.SpaceTimeGraph)
                
                #exit()
                #self.Keyprocessor.start(keypoints[0,:,:])
                #print("\n\nkeypoints shape: ",keypoints.shape)
                #print(keypoints)
                #print("\n\n")
                #exit()
                #(detecctions,keypoints,x+y
                image = self.drawPoseCOCO18(image, objhist,self.IDDict[IDkey].rank,self.IDDict[IDkey].actionBool,frame,numofOBJ)
                numofOBJ=numofOBJ+1


        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image)

    def gt_validation(self, frame, image, ObjectHistoryList):
        validated_gts = []
        gt_idxs_for_frame = np.where(self.gt[:,2]==frame)[0]
        gts_to_match = self.gt[gt_idxs_for_frame, :]

        if (len(ObjectHistoryList)>0) and (gts_to_match.shape[0]>0):
            match_table = np.zeros((len(ObjectHistoryList),gts_to_match.shape[0]))
            for o in range(len(ObjectHistoryList)):
                oh = ObjectHistoryList[o]
                for g in range(len(gts_to_match)):
                    gt = gts_to_match[g,:]
                    oh_bbox = oh.sendObject.bbox
                    gt_bbox = upscale_xywh_bbox(gt[3:], image.shape[1], image.shape[0])
                    match_iou = IoU(oh_bbox,gt_bbox)
                    if match_iou >= self.gt_match_threshold:
                        match_table[o,g] = match_iou

            best_match = match_table.max()
            while best_match >= self.gt_match_threshold:
                match_pos = np.where(match_table == best_match)
                o = match_pos[0][0]
                cam_label = ObjectHistoryList[o].sendObject.label
                g = match_pos[1][0]
                gt_label = gts_to_match[g,1]
                gts_to_match[g,0] = -1
                validated_gts.append([gt_label, cam_label])

                match_table[o,:] = np.zeros(match_table[o,:].shape)
                match_table[:,g] = np.zeros(match_table[:,g].shape)

                best_match = match_table.max()

        for gt in gts_to_match:
            if gt[0] != -1:
                validated_gts.append([gt[1], -1])

        return validated_gts

    def processFrame(self, frame):
        validated_gts = []
        self.sendQ = [] # Empty the sendQ
        if (np.where(self.validframes == frame)):
            # Fetch image and ReID features for current frame
            image, bboxes, keypoints, valid_keypoints, encodedFeatures = self.getFrameFeatures(frame)

            # Update table life values. Send expired entries to server if they were previously sent
            # to the edge server
            active_table_idxs = []
            inactive_table_idxs = []
            for idx,hist in enumerate(self.id_table):
                # Update life on previously active table elements
                if hist.life > 0:
                    hist.life -= 1
                # Evict elements that have reached end of lifespan
                if hist.life == 0:
                    if hist.sentToServer == 1:
                        # Send update to server
                        hist.sendObject.lock = 0
                        self.sendQ.append(hist.sendObject)
                    # Evict block
                    self.id_table[idx] = ObjectHistory(-1, PersonIDType())
                # Add elements to currently active/inactive lists
                if hist.life <= 0:
                    inactive_table_idxs.append(idx)
                else:
                    active_table_idxs.append(idx)
            
            # Check the recvQ for updates from edge server
            for update in self.recvQ:
                oldID = update[0]
                newID = update[1]
                newFeats = update[2]
                newFeatIdx = update[3]
                for idx,hist in enumerate(self.id_table):
                    if hist.sendObject.label == oldID:
                        hist.sendObject.label = newID
                        hist.sendObject.feats = newFeats
                        hist.sendObject.featidx = newFeatIdx
                        hist.reIDFlag = 1
                        hist.seen = self.server_seen_threshold
                        break
            self.recvQ = [] # Empty the recvQ

            ObjectHistoryList = []

            # Perform local reID
            valid_detections = np.where(valid_keypoints > self.keypoint_threshold)
            valid_detections = valid_detections[0].tolist()
            num_detections = len(valid_detections)
            num_active_table = len(active_table_idxs)
            if (num_detections>0) and (num_active_table>0):
                # Construct and fill match table
                match_table = np.full((num_detections,num_active_table), np.inf, dtype=float)
                for d in range(num_detections):
                    det_idx = valid_detections[d]
                    det_bbox = bboxes[det_idx,:]
                    det_feats = encodedFeatures[det_idx,:].reshape((1280,1))
                    for t in range(num_active_table):
                        tab_idx = active_table_idxs[t]
                        tab_object = self.id_table[tab_idx].sendObject
                        tab_bbox = tab_object.bbox
                        tab_feats = tab_object.feats
                        match_iou = self.iou_weight * (1 - IoU(det_bbox, tab_bbox))
                        match_l2norm = self.l2_weight * np.mean(la.norm((tab_feats-det_feats), ord=2, axis=0))
                        if (match_iou > 0) and (match_l2norm < self.local_match_threshold):
                            match_table[d,t] = match_iou + match_l2norm

                processed_detections = []
                best_match = match_table.min()
                while best_match < self.local_match_threshold:
                    match_pos = np.where(match_table == best_match)

                    # Fetch information for detection being entered into the table
                    d = match_pos[0][0] # Index in valid_detections list
                    det_idx = valid_detections[d]
                    det_bbox = bboxes[det_idx,:]
                    det_feats = encodedFeatures[det_idx,:].reshape((1280,1))
                    det_valid_keypoints = valid_keypoints[det_idx]
                    det_keypoints = keypoints[det_idx,:,:]
                    
                    # Fetch id entry in table
                    t = match_pos[1][0] # Index in active_table_idxs list
                    tab_idx = active_table_idxs[t]
                    # tab_hist = self.id_table[tab_idx]
                    # tab_object = tab_hist.sendObject

                    # Update identity information
                    self.id_table[tab_idx].life = self.table_life
                    self.id_table[tab_idx].seen += 1
                    self.id_table[tab_idx].sendObject.bbox = det_bbox
                    self.id_table[tab_idx].keypoints = det_keypoints

                    if (det_valid_keypoints >= self.id_table[tab_idx].keyCount):
                        if self.id_table[tab_idx].sendObject.feats.shape[1] < self.max_feats:
                            self.id_table[tab_idx].sendObject.feats = np.append(self.id_table[tab_idx].sendObject.feats,det_feats,axis=1)
                        else:
                            self.id_table[tab_idx].sendObject.feats[:,self.id_table[tab_idx].sendObject.featidx] = det_feats.reshape(1280)
                            self.id_table[tab_idx].sendObject.featidx = (self.id_table[tab_idx].sendObject.featidx+1) % self.max_feats
                        self.id_table[tab_idx].keyCount = det_valid_keypoints
                        if (det_valid_keypoints >= self.server_keypoint_threshold) and (self.id_table[tab_idx].seen >= self.server_seen_threshold):
                            self.id_table[tab_idx].sendObject.lock = 1
                            self.sendQ.append(self.id_table[tab_idx].sendObject)
                            self.id_table[tab_idx].sentToServer = 1
                            if self.opts.enable_edge_server == 0:
                                self.id_table[tab_idx].reIDFlag = 1
                    
                    ObjectHistoryList.append(self.id_table[tab_idx])

                    # Remove detection and table indices from match_table
                    match_table[d,:] = np.full(match_table[d,:].shape, np.inf, dtype=float)
                    match_table[:,t] = np.full(match_table[:,t].shape, np.inf, dtype=float)
                    processed_detections.append(d)

                    # Calculate new best match
                    best_match = match_table.min()
            
                # Clear processed detections from list of remaining valid detections
                for d in processed_detections:
                    valid_detections[d] = 'MATCHED'
                while 'MATCHED' in valid_detections:
                    valid_detections.remove('MATCHED')

            # Process any remaining detections as long as the ID table isn't full
            while (len(inactive_table_idxs)>0) and (len(valid_detections)>0):
                det_idx = valid_detections.pop(0)
                tab_idx = inactive_table_idxs.pop(0)
                sendObject = PersonIDType(self.id, self.currLabel, encodedFeatures[det_idx,:], bboxes[det_idx,:], 1)
                if self.currLabel < ((self.id + 1) * 1000000 - 1):
                    self.currLabel += 1
                else:
                    self.currLabel = self.id * 1000000
                perc_valid_keypoints = valid_keypoints[det_idx]
                self.id_table[tab_idx] = ObjectHistory(self.table_life, sendObject, perc_valid_keypoints)
                self.id_table[tab_idx].keypoints = keypoints[det_idx,:,:]
                if perc_valid_keypoints >= self.server_keypoint_threshold:
                    self.sendQ.append(sendObject)
                    self.id_table[tab_idx].sentToServer = 1
                    if self.opts.enable_edge_server == 0:
                        self.id_table[tab_idx].reIDFlag = 1

                ObjectHistoryList.append(self.id_table[tab_idx])

            #if self.opts.draw_bboxes:
                #self.drawFrameBoxes(frame, image, ObjectHistoryList)

            if self.opts.draw_poses:
                self.drawFramePoses(frame, image, ObjectHistoryList)
                #self.drawPoseCOCO18(image, ObjectHistoryList)

            if self.opts.eval_set < 2:
                validated_gts = self.gt_validation(frame, image, ObjectHistoryList)
                
                
            self.currFrameNum=self.currFrameNum+1
        return validated_gts
