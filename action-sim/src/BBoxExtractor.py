import numpy as np
import cv2

def extractBBoxes(image, frontend_net):
    bboxlist = []
    poses = frontend_net.runPoseEstimation(image)
    for pose in poses:
        # bbox has the form [x1, y1, w, h, confidence]
        bbox = np.append(pose.bbox, pose.num_found_keypoints/18)
        bboxlist.append(pose.bbox)
    bboxes = np.array(bboxlist)
    return bboxes, pose.keypoints