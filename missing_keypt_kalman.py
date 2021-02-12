
import argparse
import csv
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class Kalman():
    def __init__(self):
        self.my_filter = KalmanFilter(dim_x=2, dim_z=2)
        self.my_filter.x = np.array([[0],[0.1]])# initial state (location and velocity)
        self.my_filter.F = np.array([[1.,0.],[0.,1.]])    # state transition matrix (dim_x, dim_x)
        self.my_filter.H = np.array([[1.,0.],[0.,1.]])    # Measurement function  (dim_z, dim_x
        self.my_filter.P = np.array(np.eye(2))*1000      # initial uncertainty
        self.my_filter.R = 5                     # state uncertainty,#measurement noise vector  (dim_z, dim_z)
        self.my_filter.Q = Q_discrete_white_noise(dim=2, dt=2, var=0.5) # process uncertainty  (dim_x, dim_x)
        
    def step(self,input):
            self.my_filter.predict()
            if input[0] == -1:
                self.my_filter.update(None)
            else:
                self.my_filter.update(input)
            return np.squeeze(self.my_filter.x)

class missing_keypt_regression():
    def __init__(self,input_file, output_file, image_dir,image_output_path, gt_file):
        left_keypoints = self.ReadInputCsv(input_file)

        # Initialise the knee filter 
        Knee_filter = Kalman()
        Knee_output = np.zeros((len(left_keypoints),2),dtype = float)
        
        # Initialise the Foot filter 
        Foot_filter = Kalman()
        Foot_output = np.zeros((len(left_keypoints),2),dtype = float)

        # Initialise the Shoulder filter 
        Shoulder_filter = Kalman()
        Shoulder_ouput = np.zeros((len(left_keypoints),2),dtype = float)
        # Initialise the Hip filter
        Hip_filter =Kalman()
        Hip_output = Shoulder_ouput = np.zeros((len(left_keypoints),2),dtype = float)

        for index in left_keypoints:
            #Knee_output[index] = Knee_filter.step(np.array((left_keypoints[index]['Left Knee_x'], left_keypoints[index]['Left Knee_y']),dtype= float))
            Foot_output[index] = Foot_filter.step(np.array((left_keypoints[index]['Left foot_x'], left_keypoints[index]['Left foot_y']),dtype= float).reshape(-1,1))
            #Shoulder_ouput[index] = Shoulder_filter.step(np.array((left_keypoints[index]['Left Shoulder_x'], left_keypoints[index]['Left Shoulder_y']),dtype= float))
            #Hip_output[index] = Hip_filter.step(np.array((left_keypoints[index]['Left hip_x'], left_keypoints[index]['Left hip_y']),dtype= float))
        print(Foot_output)
        #WriteOutputCsv()

    def ReadInputCsv(self, input):
        """
        Read the input CSV with missing keypoint value of left hip, shoulder and foot

        Returns:
            dict[frame][left_shoulder_x][left_shoulder_y][left_hip_x][left_hip_y] 
                             [left_knee_x][left_knee_y][left_foot_x][left_foot_y]
        """
        keypoint_dict = {}
        with open(input, 'r') as csvfile: 
            csvreader = csv.reader(csvfile)
            for index,rows in enumerate(csvreader):
                if index != 0:
                    #keypoint_dict['Frame'] = int(rows[0])
                    keypoint_dict[int(rows[0])] = {'Left Shoulder_x' : rows[1], 
                                                   'Left Shoulder_y' : rows[2],
                                                   'Left hip_x'      : rows[3],
                                                   'Left hip_y'      : rows[4],
                                                   'Left Knee_x'     : rows[5],
                                                   'Left Knee_y'     : rows[6],
                                                   'Left foot_x'     : rows[7],
                                                   'Left foot_y'     : rows[8]
                    }
        return keypoint_dict
    
    

def get_args_parser():
    # Arguments PArser function
    parser = argparse.ArgumentParser('csv_keypoint ', add_help=False)
    parser.add_argument('--input', default=" ", type=str)
    parser.add_argument('--output', default=" ", type=str)
    parser.add_argument('--image_dir', default=" ", type=str, 
                        help='Image directory entire folder path')
    parser.add_argument('--outputimagedir', default=" ", type=str, 
                        help='Output image folder to write the images with keypoints')
    parser.add_argument('--gt_file', default=" ", type=str, 
                        help='Gt csv directory entire folder path')
    return parser

if __name__ == '__main__':

    parser = argparse.ArgumentParser('option\'s', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # instantiate the regression class
    regressor = missing_keypt_regression(input_file = args.input, output_file = args.output, 
                                    image_dir= args.image_dir,image_output_path=args.outputimagedir, gt_file = args.gt_file)

