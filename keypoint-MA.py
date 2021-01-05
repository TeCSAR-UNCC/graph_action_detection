import requests
import json
import wget
import math
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
import sys
import csv

#just the kalman filter
class kalmanFilter:
        #create the filter
	def __init__(self):
		self.my_filter = KalmanFilter(dim_x=2, dim_z=1)
		self.my_filter.x = np.array([[0],[0.1]])# initial state (location and velocity)
		self.my_filter.F = np.array([[1.,1.],[0.,1.]])    # state transition matrix (dim_x, dim_x)
		self.my_filter.H = np.array([[1.,0.]])    # Measurement function  (dim_z, dim_x)
		self.my_filter.P *= 1000.                 # covariance matrix ( dim x, dim x) 
		self.my_filter.R = 5                      # state uncertainty,#measurement noise vector  (dim_z, dim_z)
		self.my_filter.Q = Q_discrete_white_noise(dim=2, dt=2, var=0.5)# process uncertainty  (dim_x, dim_x)
	
	def step(self,angle):
		self.my_filter.predict()
		self.my_filter.update(angle)
		return self.my_filter.x



def leadZero(x):
	y=''
	#temp=''
	for i in range(6-len(str(x))):
		y='0'+y
	y=y+str(x)
	return y


endString='.jpg'


#this file is the save file for the regressed hip angles:
HIPansfile="/home/justin/Documents/mobility/other/hip-ans-new.csv"

##this file is the save file for the regressed knee angles::
Kneeansfile="/home/justin/Documents/mobility/other/knee-ans-new.csv"

#this file is for recording which images give issues with reading keypoints:
failfile="/home/justin/Documents/mobility/other/MA-issues.csv"

LastHipInfrence=0
HIPanswer=0
Knee_Answer=0
lastKnee=0
HIP_TryList=[]
KNEE_TryList=[]
fail_list=[]
kfilter = kalmanFilter()
kneefilter = kalmanFilter()

for i in range(175):
    #input src of video
	sourceFile='/home/justin/Documents/mobility/camera3/'
	fail=0
	fileNum=leadZero(i+1)
	sourceFile=sourceFile+fileNum+endString

	print(sourceFile)
	
	r = requests.post(
	    "https://api.deepai.org/api/pose-detection",
	    files={
		'image': open(sourceFile, 'rb'),
	    },
	    headers={'api-key': '6b5b64b2-cfc0-4e31-960c-ae34432a8c40'}
	)
	#print(r.json())
	#exit()
	out = json.loads(r.text)
	part=['_shoulder','_hip','_knee','_foot']
	leftside_point=[]
	rightside_point=[]
	for i in range(4):
		try:
			temp=out['output']['people'][0]['pose']['left'+part[i]]
			hasValue=temp[0]
			leftside_point.append(temp)
		except:
			temp=[-1,-1]
			#fail=1
			leftside_point.append(temp)
		
		try:
			temp=out['output']['people'][0]['pose']['right'+part[i]]
			hasValue=temp[0]#test to see if its defined
			rightside_point.append(temp)
		except:
			temp=[-1,-1]
			#fail=1
			rightside_point.append(temp)
	#print(rightside_point)
	#print(leftside_point)
	#print(rightside_point[0])
	#print(rightside_point[0][0])
	#exit()
	

	#keypoint parser and angle infrence:
	if (rightside_point[0][0] != -1) and (rightside_point[1][0] != -1) and (rightside_point[2][0] != -1) and (rightside_point[3][0] != -1):
		sho_R=rightside_point[0]
		hip_R=rightside_point[1]
		knee_R=rightside_point[2]
		foot_R=rightside_point[3]
		#hip
		A= math.sqrt( ((sho_R[0]-hip_R[0])**2)+ ((sho_R[1]-hip_R[1])**2) )
		B= math.sqrt( ((hip_R[0]-knee_R[0])**2)+ ((hip_R[1]-knee_R[1])**2) )
		C= math.sqrt( ((sho_R[0]-knee_R[0])**2)+ ((sho_R[1]-knee_R[1])**2) )
		#knee
		D= math.sqrt( ((hip_R[0]-knee_R[0])**2)+ ((hip_R[1]-knee_R[1])**2) )
		E= math.sqrt( ((knee_R[0]-foot_R[0])**2)+ ((knee_R[1]-foot_R[1])**2) )
		F= math.sqrt( ((hip_R[0]-foot_R[0])**2)+ ((hip_R[1]-foot_R[1])**2) )
		try:
			HIPanswer=180-math.degrees(math.acos( ( (C**2)-(A**2)-(B**2) )/(-2*A*B) ) )
			LastHipInfrence=HIPanswer
			executeUpdateHip=kfilter.step(LastHipInfrence)[0]
		except:
			HIPanswer =float(kfilter.step(LastHipInfrence)[0])
			fail=1
		try:
			Knee_Answer=180-math.degrees(math.acos( ( (F**2)-(D**2)-(E**2) )/(-2*D*E) ) )
			lastKnee=Knee_Answer
			executeUpdateKnee=kneefilter.step(lastKnee)[0]
		except:
			Knee_Answer =float(kneefilter.step(lastKnee)[0])
			fail=2
			
	elif (leftside_point[0][0] != -1) and (leftside_point[1][0] != -1) and (leftside_point[2][0] != -1) and (leftside_point[3][0] != -1):
		sho_L=leftside_point[0]
		hip_L=leftside_point[1]
		knee_L=leftside_point[2]
		foot_L=leftside_point[3]
		
		A= math.sqrt( ((sho_L[0]-hip_L[0])**2)+ ((sho_L[1]-hip_L[1])**2) )
		B= math.sqrt( ((hip_L[0]-knee_L[0])**2)+ ((hip_L[1]-knee_L[1])**2) )
		C= math.sqrt( ((sho_L[0]-knee_L[0])**2)+ ((sho_L[1]-knee_L[1])**2) )
		
		
		D= math.sqrt( ((hip_L[0]-knee_L[0])**2)+ ((hip_L[1]-knee_L[1])**2) )
		E= math.sqrt( ((knee_L[0]-foot_L[0])**2)+ ((knee_L[1]-foot_L[1])**2) )
		F= math.sqrt( ((hip_L[0]-foot_L[0])**2)+ ((hip_L[1]-foot_L[1])**2) )
		try:
			HIPanswer=180-math.degrees(math.acos( ( (C**2)-(A**2)-(B**2) )/(-2*A*B) ) )
			LastHipInfrence=HIPanswer
			executeUpdate=kfilter.step(LastHipInfrence)[0]
		except:
			HIPanswer =float(kfilter.step(LastHipInfrence)[0])
			LastHipInfrence=HIPanswer
			fail=1

		try:
			Knee_Answer=180-math.degrees(math.acos( ( (F**2)-(D**2)-(E**2) )/(-2*D*E) ) )
			lastKnee=Knee_Answer
			executeUpdateKnee=kneefilter.step(lastKnee)[0]
		except:
			Knee_Answer =float(kneefilter.step(lastKnee)[0])
			fail=2
	else:
		HIPanswer =float(kfilter.step(LastHipInfrence)[0])
		LastHipInfrence=HIPanswer
		
		Knee_Answer =float(kneefilter.step(lastKnee)[0])
		lastKnee=Knee_Answer
		
		fail=3
		
	HIP_TryList.append(HIPanswer)
	KNEE_TryList.append(Knee_Answer)
	fail_list.append(fail)
	
		

with open(HIPansfile, mode='w') as Pred_angle:
	Pred_angle_writer = csv.writer(Pred_angle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(HIP_TryList)):
		Pred_angle_writer.writerow([HIP_TryList[i]])
		
		
with open(Kneeansfile, mode='w') as Pred_angle:
	Pred_angle_writer = csv.writer(Pred_angle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(KNEE_TryList)):
		Pred_angle_writer.writerow([KNEE_TryList[i]])
		
with open(failfile, mode='w') as Pred_angle:
	Pred_angle_writer = csv.writer(Pred_angle, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(fail_list)):
		Pred_angle_writer.writerow([fail_list[i]])
