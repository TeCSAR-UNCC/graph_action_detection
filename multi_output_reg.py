"""
Script to predict missing keypoints - Data imputation using Random forest regression
"""
# import libraries
import  argparse
import csv
import numpy as np
import math
import os
import cv2
from os.path import isfile, join

# Import regression libraries from sklearn
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV #For hyperparameter tuning

# Import kalman filter libraries
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Reproduciblity
SEED = 2

class kalmanFilter:
    #create the kalman filter
	def __init__(self):
		self.my_filter = KalmanFilter(dim_x=2, dim_z=1)
		self.my_filter.x = np.array([[0],[0.1]])   
		self.my_filter.F = np.array([[1.,1.],[0.,1.]])     # state transition matrix (dim_x, dim_x)
		self.my_filter.H = np.array([[1.,0.]])              # Measurement function  (dim_z, dim_x)
		self.my_filter.P *= 30.                             # covariance matrix ( dim x, dim x) 
		self.my_filter.R = 5                      
		self.my_filter.Q = Q_discrete_white_noise(dim=2, dt=2, var=0.5) #process uncertainty  (dim_x, dim_x)
	
	def step(self,angle):
        # Update the angles smoothen them out
		self.my_filter.predict()
		self.my_filter.update(angle)
		return self.my_filter.x

class keypoint_regression():
    """
    Class to regress the missing keypoint
    """
    def __init__(self, input_file, output_file, image_dir, image_output_path ,gt_file, X1ColumnName, X2ColumnName ,
                Y1ColumnName, Y2ColumnName):
        super().__init__()
        
        
        # Read all the keypoint values and columns 
        self.left_keypoints = self.ReadInputCsv(input_file)
        self.outputFile = output_file

        # Seperate the X dependent variable and Y target varaibles from others from keypoint dict
        XTrainVariables, YTrainVariables, XTestVariables, YTestVariables = self.SeperateXYVariables(X1ColumnName, X2ColumnName ,
                Y1ColumnName, Y2ColumnName)
        assert len(XTrainVariables) == len(YTrainVariables)
        assert len(XTestVariables) == len(YTestVariables)

        # Perform multi output regression on the train and test set to predict the missing values
        missing_values, X_test = self.MultiOutputRegression(XTrainVariables, YTrainVariables, XTestVariables, YTestVariables)

        # Update the missing values into the keypoint dictionary
        self.UpdateMissing(X_test, missing_values, ColumnNameX = (X1ColumnName,X2ColumnName), ColumnNameY = (Y1ColumnName, Y2ColumnName))

        # Calculate the angle for knee bend and hip bend
        self.CalculateAngles()

        # Write the result back into csv with two new columns knee bend and hip bend
        self.WriteKeypointCsv()

        # Draw the keypoints regressed on the images to visualize
        self.drawPose(image_dir,image_output_path, gt_file) 

        # Convert the frames to video
        self.ImageToVideo()

    def WriteKeypointCsv(self):
        
        """
        Write Keypoints back to CSV with two new columns Knee bend and Hip bend with 
        calculated angles 
        """
        
        fields = ['Left Shoulder_x', 'Left Shoulder_y', 'Left hip_x',
                                                   'Left hip_y',
                                                   'Left Knee_x',
                                                   'Left Knee_y',
                                                   'Left foot_x',
                                                   'Left foot_y',
                                                   'Knee_bend',
                                                   'Hip_bend']
        
        with open(self.outputFile, 'w') as csvfile:  
         # creating a csv dict writer object  
            writer = csv.DictWriter(csvfile, fieldnames = fields)  
        
            # writing headers (field names)  
            writer.writeheader()  

            for i in range(len(self.left_keypoints)):
                # writing data rows  
                writer.writerow(self.left_keypoints[i])  
        print("CSV write Done")

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

    def drawPose(self, image_dir, image_output_path, gt_file):
        """
        Pull Keypoints on to the Image
        """
        # Read the groudtruth csv and store it in a list
        self.KneeGT = []
        self.HipGT = []
        with open(gt_file, 'r') as read_obj:
                csv_reader = csv.reader(read_obj)
                for index,row in enumerate(csv_reader):
                    if index !=0 :
                        self.KneeGT.append(float(row[1]))
                        self.HipGT.append(float(row[2]))

        for filename in sorted(os.listdir(image_dir)):
            
            imageFilePath = image_dir + '/' + filename
            image = cv2.imread(imageFilePath)

            Frame = int(filename.split('.jpg')[0])
            
            LeftKneeCoord = (int(float(self.left_keypoints[Frame]['Left Knee_x'])),int(float(self.left_keypoints[Frame]['Left Knee_y'])))
            LeftHipCoord = (int(float(self.left_keypoints[Frame]['Left hip_x'])),int(float(self.left_keypoints[Frame]['Left hip_y'])))
            LeftShoulderCoord = (int(float(self.left_keypoints[Frame]['Left Shoulder_x'])),int(float(self.left_keypoints[Frame]['Left Shoulder_y'])))
            LeftFootCoord = (int(float(self.left_keypoints[Frame]['Left foot_x'])),int(float(self.left_keypoints[Frame]['Left foot_y'])))

            # Draw the circle at keypoints coordinates
            image = cv2.circle(image, (LeftKneeCoord), 12, (0,0,255))
            image = cv2.circle(image, (LeftHipCoord), 12, (0,0,255))
            image = cv2.circle(image, (LeftShoulderCoord), 12, (0,0,255))
            image = cv2.circle(image, (LeftFootCoord), 12, (0,0,255))

            # Connect the keypoint pair for visuals
            image = cv2.line(image, (LeftShoulderCoord), (LeftHipCoord), (0,0,255), 6)
            image = cv2.line(image, (LeftHipCoord), (LeftKneeCoord), (0,0,255), 6)
            image = cv2.line(image, (LeftKneeCoord), (LeftFootCoord), (0,0,255), 6)

            # Add the gt and results of knee bend and hip bend 
            scale = 0.7

            # Knee bend Values display
            Knee_gt = "Knee bend motion capture : " + str(self.KneeGT[Frame])[:8] + " degs"
            txt_size = cv2.getTextSize(Knee_gt, cv2.FONT_HERSHEY_SIMPLEX, scale, cv2.FILLED)
            end_x = 20 + txt_size[0][0] 
            end_y = 20 - txt_size[0][1] 
            image = cv2.rectangle(image, (20,20), ((end_x+200), (end_y+200)), (255, 255, 255), cv2.FILLED)
            Knee_text = "Knee bend AI : " + str(self.left_keypoints[Frame]['Knee_bend'])[:8] + " degs"
            image = cv2.putText(image, Knee_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, scale*1.2, (0,0,255), 2)
            image = cv2.putText(image, Knee_gt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, scale*1.2, (255,0,0), 2)

            
            # Hip bend values display
            Hip_text = "Hip bend AI : " + str(self.left_keypoints[Frame]['Hip_bend'])[:8] + " degs"
            image = cv2.putText(image, Hip_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, scale*1.2, (0,0,255), 2)
            Hip_gt = "Hip bend motion capture : " + str(self.HipGT[Frame])[:8] + "degs"
            image = cv2.putText(image, Hip_gt, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, scale*1.2, (255,0,0), 2)

            Frame = "Frame number : " + str(Frame)
            image = cv2.putText(image, str(Frame), (20, 170), cv2.FONT_HERSHEY_SIMPLEX, scale*1.2, (0,255,0), 2)
            filename = image_output_path +  '/' + filename
            cv2.imwrite(filename, image)
        print("Finished writing images to the directory with keypoints")

    def SeperateXYVariables(self, X1ColumnName, X2ColumnName ,
                Y1ColumnName, Y2ColumnName):
        """
        Function to get only columns from the keypoint dict and remove the missing 
        values

        Returns : X_train_array [n_samples, 2]
                  Y_train_array [n_samples, 2]
                  X_test_array [missing_nsamples, 2]
                  Y_test_array [missing_nsamples, 2]
        """
        X_train_array = []
        Y_train_array = []
        X_test_array = []
        Y_test_array = []

        
        for frame,values in (self.left_keypoints.items()):
           if (float(self.left_keypoints[frame][Y1ColumnName]) != -1) and (float(self.left_keypoints[frame][Y2ColumnName]) != -1):
            if (float(self.left_keypoints[frame][X1ColumnName]) > 0) and (float(self.left_keypoints[frame][X2ColumnName]) > 0):
                X_train_array.append((self.left_keypoints[frame][X1ColumnName],self.left_keypoints[frame][X2ColumnName]))
                Y_train_array.append((self.left_keypoints[frame][Y1ColumnName],self.left_keypoints[frame][Y2ColumnName]))
           else:
               X_test_array.append((self.left_keypoints[frame][X1ColumnName],self.left_keypoints[frame][X2ColumnName]))
               Y_test_array.append((self.left_keypoints[frame][Y1ColumnName],self.left_keypoints[frame][Y2ColumnName]))
        
        X_train_array = np.asarray((X_train_array),dtype = np.float)
        Y_train_array = np.asarray(Y_train_array,dtype = np.float)
        X_test_array = np.asarray(X_test_array,dtype = np.float)
        Y_test_array = np.asarray(Y_test_array,dtype = np.float)
        return X_train_array, Y_train_array, X_test_array, Y_test_array

    def MultiOutputRegression(self, XTrainVariables, YTrainVariables, XTestVariables, YTestVariables):
        """
        Perform random forest multi output regression on the features to predict the missing

        Returns : y_test : 2D array [n_missingsamples, 2]
                 XTestVariables : 2D array [n_missingsamples_xfeature, 2]
        """
        
        # Train Test split
        X_rftrain, X_rfval, y_rftrain, y_rfval = train_test_split(XTrainVariables, YTrainVariables, test_size=0.2, random_state=SEED)

        #params = self.hyper_parameter_tuning(X_rftrain, y_rftrain, XTestVariables)
        
        # Random forest
        rf = RandomForestRegressor(random_state=SEED, n_estimators = 10, min_samples_leaf = 2)

        # Run this if hyper parameter tuning is done
        #rf = RandomForestRegressor(bootstrap = params['bootstrap'], random_state=SEED, n_estimators=params['n_estimators'], 
        #                        min_samples_leaf = params['min_samples_leaf'] , max_features= params['max_features'],
        #                        max_depth= params['max_depth'], min_samples_split=params['min_samples_split'])
    

        # Fit 'rf' to the training set
        rf.fit(X_rftrain, y_rftrain)
        
        # Predict the test set labels 'y_rfpred'
        y_rfpred = rf.predict(X_rfval)

        # Test it on the train dataset too
        y_rftrain_hat = rf.predict(X_rftrain)

        # Evaluate the train set RMSE
        rmse_rftrain = MSE(y_rftrain, y_rftrain_hat)**(1/2)
        
        # Evaluate the val set RMSE
        rmse_rfval = MSE(y_rfval, y_rfpred)**(1/2)
        
        # Print the test set RMSE
        print('Train set RMSE of Random Forest regressor: {} '.format(rmse_rftrain))
        print('Validation set RMSE of Random Forest regressor: {} '.format(rmse_rfval))

        # Predict the values of the Test data set
        y_test = rf.predict(XTestVariables)

        return y_test, XTestVariables
    
    def hyper_parameter_tuning(self, X_train, y_train, XTestVariables):
        """
        Perform hyper parameter tuning using randomisedsearch cv
        """
        
        # Number of trees in random forest 
        n_estimators = [int(x) for x in np.linspace(start = 2, stop = 20, num = 10)] 
        
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']

        # Maximum number of levels in tree : Creates this array : [5, 10, 15, 20, 25]
        max_depth = [int(x) for x in np.linspace(5, 25, num = 5)]
        max_depth.append(None) #Add none to the array

        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4, 8]

        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestRegressor()

        # Random search of parameters, using 3 fold cross validation, 
        # search across 60 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=SEED, n_jobs = -1)

        # Fit the random search model
        rf_random.fit(X_train, y_train)

        # Predict the values of the Test data set
        y_test = rf_random.predict(XTestVariables)

        return rf_random.best_params_

    def UpdateMissing(self, X_test, missing_values, ColumnNameX, ColumnNameY):
        
        """
        Update the missing values (-1) with regressed value in the keypoints dict
        """
        count = 0
        for i,values in enumerate(X_test):
            for index in sorted(self.left_keypoints):
                if (float(self.left_keypoints[index][ColumnNameX[0]]) ==values[0]) and (float(self.left_keypoints[index][ColumnNameX[1]]) == values[1]) and (float(self.left_keypoints[index][ColumnNameY[0]]) == -1.0):
                    self.left_keypoints[index][ColumnNameY[0]] = str(missing_values[i][0])
                    self.left_keypoints[index][ColumnNameY[1]] = str(missing_values[i][1])
                    count += 1
                    break
        print("Total missing values calculated : {}".format(count))

    def CalculateAngles(self):
        """
        Calculate the missing angles for Knee bend and Hip bend from the keypoint dict 
        and update the values in the keypoint dict with key Knee_bend, Hip_bend
        """
        Kneefilter = kalmanFilter()
        Hipfilter = kalmanFilter()
        
        # calculates Angles for Knee bend
        for frame,values in self.left_keypoints.items():
            A = math.sqrt( ((float(self.left_keypoints[frame]['Left hip_x']) - float(self.left_keypoints[frame]['Left Knee_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left hip_y']) - float(self.left_keypoints[frame]['Left Knee_y']))**2) ) 
            B = math.sqrt( ((float(self.left_keypoints[frame]['Left Knee_x']) - float(self.left_keypoints[frame]['Left foot_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left Knee_y']) - float(self.left_keypoints[frame]['Left foot_y']))**2) ) 
            C = math.sqrt( ((float(self.left_keypoints[frame]['Left hip_x']) - float(self.left_keypoints[frame]['Left foot_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left hip_y']) - float(self.left_keypoints[frame]['Left foot_y']))**2) ) 
            
            LastKneeInference = 180-math.degrees(math.acos( ( (C**2)-(A**2)-(B**2) )/(-2*A*B) ) )
            self.left_keypoints[frame]['Knee_bend'] = Kneefilter.step(LastKneeInference)[0][0]
            # calculate Angle for hip bend
            D = math.sqrt( ((float(self.left_keypoints[frame]['Left Shoulder_x']) - float(self.left_keypoints[frame]['Left hip_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left Shoulder_y']) - float(self.left_keypoints[frame]['Left hip_y']))**2) ) 
            E = math.sqrt( ((float(self.left_keypoints[frame]['Left hip_x']) - float(self.left_keypoints[frame]['Left Knee_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left hip_y']) - float(self.left_keypoints[frame]['Left Knee_y']))**2) ) 
            F = math.sqrt( ((float(self.left_keypoints[frame]['Left Shoulder_x']) - float(self.left_keypoints[frame]['Left Knee_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left Shoulder_y']) - float(self.left_keypoints[frame]['Left Knee_y']))**2) ) 
            
            LastHipInference = 180-math.degrees(math.acos( ( (F**2)-(D**2)-(E**2) )/(-2*D*E) ) )
            self.left_keypoints[frame]['Hip_bend'] = Hipfilter.step(LastHipInference)[0][0]
    
    def ImageToVideo(self):


        frame_folder = "/mnt/AI_RAID/Mobility-assessment/Output_frames/RandomForest_FinalImages/" 
        img_array = []
        files = [f for f in os.listdir(frame_folder) if isfile(join(frame_folder, f))]
        #for sorting the file names properly
        files.sort(key = lambda x: x[5:-4])
        files.sort()
        
        for filename in files:
            image_file = frame_folder + filename
            img = cv2.imread(image_file)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        
        out = cv2.VideoWriter('Result.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        print("Finished writing images to the video with keypoints")


def get_args_parser():
    # Arguments PArser function
    parser = argparse.ArgumentParser('csv_keypoint ', add_help=False)
    parser.add_argument('--input', default=" ", type=str)
    parser.add_argument('--output', default=" ", type=str)
    parser.add_argument('--X1', default=" ", type=str)
    parser.add_argument('--X2', default=" ", type=str)
    parser.add_argument('--Y1', default=" ", type=str)
    parser.add_argument('--Y2', default=" ", type=str)
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
    regressor = keypoint_regression(input_file = args.input, output_file = args.output, 
                                    image_dir= args.image_dir, image_output_path=args.outputimagedir, gt_file = args.gt_file,
                                    X1ColumnName = args.X1, X2ColumnName = args.X2, Y1ColumnName = args.Y1,
                                    Y2ColumnName = args.Y2)
