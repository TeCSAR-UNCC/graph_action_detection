import os
import re
import shutil
import numpy as np


Log1File='/home/justin/Documents/new-research/keypoint-new/st-gcn-data-len-master/work_dir/logs/log_1.txt'
Log2File='/home/justin/Documents/new-research/keypoint-new/st-gcn-data-len-master/work_dir/logs/log_2.txt'
top1ListL1=[]
pattern='^.*Top1.*$'
# Using readlines() 
file1 = open('Log1File', 'r') 
#Lines = file1.readlines() 
  
while True: 
    count += 1
  
    # Get next line from file 
    line = file1.readline() 
    TBool=re.match(pattern, line)
    if TBool:
    	top1ListL1.append(line)
  
    # if line is empty 
    # end of file is reached 
    if not line: 
        break
       


file1.close() 
