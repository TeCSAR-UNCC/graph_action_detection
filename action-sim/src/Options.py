import argparse

from argparse import ArgumentTypeError as err
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
    
class PathType(object):
    def __init__(self, exists=True, type='file', dash_ok=True):
        '''exists:
                True: a path that does exist
                False: a path that does not exist, in a valid parent directory
                None: don't care
           type: file, dir, symlink, None, or a function returning True for valid paths
                None: don't care
           dash_ok: whether to allow "-" as stdin/stdout'''

        assert exists in (True, False, None)
        assert type in ('file','dir','symlink',None) or hasattr(type,'__call__')

        self._exists = exists
        self._type = type
        self._dash_ok = dash_ok

    def __call__(self, string):
        if string=='-':
            # the special argument "-" means sys.std{in,out}
            if self._type == 'dir':
                raise err('standard input/output (-) not allowed as directory path')
            elif self._type == 'symlink':
                raise err('standard input/output (-) not allowed as symlink path')
            elif not self._dash_ok:
                raise err('standard input/output (-) not allowed')
        else:
            e = os.path.exists(string)
            if self._exists==True:
                if not e:
                    raise err("path does not exist: '%s'" % string)

                if self._type is None:
                    pass
                elif self._type=='file':
                    if not os.path.isfile(string):
                        raise err("path is not a file: '%s'" % string)
                # elif self._type=='symlink':
                #     if not os.path.symlink(string):
                #         raise err("path is not a symlink: '%s'" % string)
                elif self._type=='dir':
                    if not os.path.isdir(string):
                        raise err("path is not a directory: '%s'" % string)
                elif not self._type(string):
                    raise err("path not valid: '%s'" % string)
            else:
                if self._exists==False and e:
                    raise err("path exists: '%s'" % string)

                p = os.path.dirname(os.path.normpath(string)) or '.'
                if not os.path.isdir(p):
                    raise err("parent path is not a directory: '%s'" % p)
                elif not os.path.exists(p):
                    raise err("parent directory does not exist: '%s'" % p)

        return string


def ActionGet_parser():

    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('-c', '--cameras', type=eval, default=(1,2,3,4,5,6,7,8), help='Cameras to run evaluation on')
    parser.add_argument('-r', '--source_framerate', type=int, default=60, help='Source camera framerate')
    parser.add_argument('-f', '--framerate', type=int, default=60, help='System framerate')
    parser.add_argument('-i', '--image_path', type=PathType(exists=True, type='dir'), help='Base directory for input images', required=True)
    parser.add_argument('-o', '--image_output_path', help='Base directory for output images')
    parser.add_argument('-j', '--json_path', help='Path for keypoint json files')
    parser.add_argument('-b', '--draw_bboxes', type=bool, help='Draw bounding boxes on frames and write to image_output_path')
    parser.add_argument('-p', '--draw_poses', type=bool, help='Draw poses on frames and write to image_output_path')
    parser.add_argument('-g', '--ground_truth', type=PathType(exists=True, type='file'), help='Ground truth file')
    parser.add_argument('-s', '--sequence', type=eval, default=(-1,), help='Frame sequence for evaluation')
    parser.add_argument('-e', '--eval_set', type=int, default=0, help='Dataset to evaluate (0-DukeMTMC, 1-DukeMTMC-OP, 2-CPCC, 3-CPCC-OP)')
    parser.add_argument('--use_gpu', type=eval, default=-1, help="Use CUDA GPU index for DNN tasks (-1 to use CPU)")
    parser.add_argument('--openpose_path', help='Path to OpenPose binary for generating keypoints')
    parser.add_argument('--enable_edge_server', type=int, default=1, help='Enable edge server for inter-camera reID')
    parser.add_argument('--pose_cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seg_cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
                        
 #aCTION                       
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument(
        '--config',
        default='./config/NTU-RGB-D/xview/ST_GCN.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=128,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')

    return parser

def parseOptions():
    parser = argparse.ArgumentParser(description='Edge video analytics test suite options', add_help=True)
    parser.add_argument(
        '--config',
        default='./config/NTU-RGB-D/xview/ST_GCN.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument('-c', '--cameras', type=eval, default=(1,2,3,4,5,6,7,8), help='Cameras to run evaluation on')
    parser.add_argument('-r', '--source_framerate', type=int, default=60, help='Source camera framerate')
    parser.add_argument('-f', '--framerate', type=int, default=60, help='System framerate')
    parser.add_argument('-i', '--image_path', type=PathType(exists=True, type='dir'), help='Base directory for input images', required=True)
    parser.add_argument('-o', '--image_output_path', help='Base directory for output images')
    parser.add_argument('-j', '--json_path', help='Path for keypoint json files')
    parser.add_argument('-b', '--draw_bboxes', type=bool, help='Draw bounding boxes on frames and write to image_output_path')
    parser.add_argument('-p', '--draw_poses', type=bool, help='Draw poses on frames and write to image_output_path')
    parser.add_argument('-g', '--ground_truth', type=PathType(exists=True, type='file'), help='Ground truth file')
    parser.add_argument('-s', '--sequence', type=eval, default=(-1,), help='Frame sequence for evaluation')
    parser.add_argument('-e', '--eval_set', type=int, default=0, help='Dataset to evaluate (0-DukeMTMC, 1-DukeMTMC-OP, 2-CPCC, 3-CPCC-OP)')
    parser.add_argument('--use_gpu', type=eval, default=-1, help="Use CUDA GPU index for DNN tasks (-1 to use CPU)")
    parser.add_argument('--openpose_path', help='Path to OpenPose binary for generating keypoints')
    parser.add_argument('--enable_edge_server', type=int, default=1, help='Enable edge server for inter-camera reID')
    parser.add_argument('--pose_cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seg_cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    

    return parser.parse_args()
