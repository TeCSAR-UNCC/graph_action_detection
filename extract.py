import cv2
import argparse

#python3 extract.py --input /home/justin/Documents/graph_action_detection/test/vid.avi  --output /home/justin/Documents/graph_action_detection/out/


def get_args_parser():
    parser = argparse.ArgumentParser('video ', add_help=False)
    parser.add_argument('--input', default=" ", type=str)
    parser.add_argument('--output', default=" ", type=str)
    return parser



parser = argparse.ArgumentParser('option\'s', parents=[get_args_parser()])
args = parser.parse_args()
inputfile=args.input
outputFilePath=args.output

print('\n input: ',inputfile,'\n')
vidcap = cv2.VideoCapture(inputfile)
success,image = vidcap.read()
count = 0
while count <5:
  name=outputFilePath+str(count).zfill(6)+'.jpg'
  print(name)
  cv2.imwrite( name, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
