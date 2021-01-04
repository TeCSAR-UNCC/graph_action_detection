import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm
import cv2
import numpy as np

from src.EfficientHRNetKeypoints.EfficientHRNet import get_pose_net
from src.EfficientHRNetKeypoints.config import pose_cfg
from src.EfficientHRNetKeypoints.group import HeatmapParser
from src.EfficientHRNetKeypoints.utils.transforms import get_final_preds
from src.EfficientHRNetKeypoints.utils.inference import get_outputs, aggregate_results
from src.EfficientHRNetKeypoints.utils.transforms import resize
from src.EfficientHRNetKeypoints.utils.vis import save_valid_image

'''
def infer_efficient(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()
    stages_output = net(tensor_img)
    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad
'''

class EfficientHRNetKeypoints():
    def __init__(self, pose_cfg):
        self.pose_cfg = pose_cfg
        self.net = get_pose_net(pose_cfg, is_train=False)
    def runPoseEstimation(self, image):
        parser = HeatmapParser(pose_cfg)
        cudnn.benchmark = pose_cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = pose_cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = pose_cfg.CUDNN.ENABLED
        
        self.net.load_state_dict(torch.load(pose_cfg.TEST.MODEL_FILE, map_location='cpu'), strict=True)
        self.net.eval()
        #self.net.cuda()
        #final_heatmaps = None
        #tags_list = []
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )#,
                #torchvision.transforms.Resize(cfg.DATASET.INPUT_SIZE)
            ]
        )
        image_resized, center, scale = resize(
                        image, pose_cfg.DATASET.INPUT_SIZE
                    )
        image_resized = transforms(image_resized)
        image_resized = image_resized.unsqueeze(0)#.cuda()

        outputs, heatmaps, tags = get_outputs(
                        pose_cfg, self.net, image_resized, with_flip=False,
                        project2image=pose_cfg.TEST.PROJECT2IMAGE
                    )
        grouped, scores = parser.parse(
                    heatmaps, tags, pose_cfg.TEST.ADJUST, pose_cfg.TEST.REFINE
                )
        final_results = get_final_preds(
                    grouped, center, scale,
                    [heatmaps.size(3), heatmaps.size(2)]
                )
        #filters redundant poses
        
        final_pts = []
        for i in range(len(final_results)):
            final_pts.insert(i,list())
            for pts in final_results[i]:
                if len(final_pts[i]) > 0:
                    diff = np.mean(np.abs(np.array(final_pts[i])[...,:2] - pts[...,:2])) 
                    if np.any(diff < 3):
                        final_pts[i].append([-1,-1,pts[2],pts[3]])
                        continue
                final_pts[i].append(pts)
        final_results = final_pts
        for idx in range(len(final_results)):
            final_results[idx] = np.concatenate(final_results[idx],axis=0)
            final_results[idx] = np.reshape(final_results[idx], (-1,4))
        #print("Final results after filter", len(final_results))
        
        keypoints = []
        h_scores = []
        x_coordinates = []
        y_coordinates = []
        for idx in range(len(final_results)):
            key_temp = []
            x_temp = []
            y_temp = []
            h_temp = []
            for i in range(len(final_results[idx])):
                keypoint = final_results[idx][i,:2]
                key_temp.append(keypoint)
                x_coor = final_results[idx][i,0]
                x_temp.append(x_coor)
                y_coor = final_results[idx][i,1]
                y_temp.append(y_coor)
                h_score = final_results[idx][i,2]
                h_temp.append(h_score)

            keypoints.append(key_temp)
            x_coordinates.append(x_temp)
            y_coordinates.append(y_temp)
            h_scores.append(h_temp)
            keypoints[idx] = np.concatenate(keypoints[idx],axis=0)
            keypoints[idx] = np.reshape(keypoints[idx], (-1,2))
        return keypoints, x_coordinates, y_coordinates, h_scores