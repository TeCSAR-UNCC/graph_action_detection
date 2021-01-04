import os

def rename_frame(frame_dir):
    count = 1
    for in_frame in sorted(os.listdir(frame_dir)):
        new_name = str(count).zfill(5)+'.jpg'
        os.rename(frame_dir+in_frame,frame_dir+new_name)
        print(new_name)
        count+=1


rename_frame('/home/justin/Desktop/seg-neff-dev/output/pa-iou-fix/camera3/')

