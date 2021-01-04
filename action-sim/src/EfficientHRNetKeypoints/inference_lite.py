from EfficientHRNet import get_pose_net
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm
import cv2
from config import cfg
from config import check_config
from config import update_config
from group import HeatmapParser
from utils.transforms import get_final_preds
from utils.inference import get_outputs
import numpy as np
from utils.transforms import resize
from utils.vis import save_valid_image
import os

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Inference EfficientHRNet Pose')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--image',
                        help='path to image for inference',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)
    parser = HeatmapParser(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = get_pose_net(cfg, is_train=False)
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    model.eval()
    model.cuda()

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
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized, center, scale = resize(
                    image, cfg.DATASET.INPUT_SIZE
                )
    image_resized = transforms(image_resized)
    image_resized = image_resized.unsqueeze(0).cuda()

    outputs, heatmaps, tags = get_outputs(
                    cfg, model, image_resized, with_flip=False,
                    project2image=cfg.TEST.PROJECT2IMAGE
                )

    grouped, scores = parser.parse(
                heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )
    final_results = get_final_preds(
                grouped, center, scale,
                [heatmaps.size(3), heatmaps.size(2)]
            )
    final_output_dir = "/1TB/EfficientHRNet-Keypoints/Lightweight_Inference/outputs"
    prefix = '{}_{}'.format(os.path.join(final_output_dir, 'result_valid'), 2)
    # logger.info('=> write {}'.format(prefix))
    save_valid_image(image, final_results, '{}.jpg'.format(prefix), dataset='COCO')
    # save_debug_images(cfg, image_resized, None, None, outputs, prefix)
    print("Finished.")

if __name__ == '__main__':
    main()