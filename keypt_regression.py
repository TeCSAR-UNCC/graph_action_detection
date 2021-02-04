# Script to regress missing keypoints

import  argparse
import csv
import numpy as np
import math
import os
import cv2

# Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV #For hyperparameter tuning
from sklearn.metrics import mean_squared_error as MSE

# importing libraries for polynomial transform
from sklearn.preprocessing import PolynomialFeatures
# for creating pipeline
from sklearn.pipeline import Pipeline

# Seed for reproduciblity
SEED = 1

class keypoint_regression():
    def __init__(self, input_file, output_file, AngleCalc, image_dir, image_output_path ,gt_file):
        super().__init__()
        
        
        # Read all the keypoint values and columns 
        self.left_keypoints = self.ReadInputCsv(input_file)
        
        # In my data the missing values are mostly foot and knee so X are hip,knee and Y is foot for knee bend angle
        # change the X and Y according to the input csv
        self.inputColumns = { 1 :[['Left hip_x', 'Left Knee_x', 'Left foot_x'],['Left hip_y', 'Left Knee_y', 'Left foot_y']],
                          2 : [['Left Shoulder_x', 'Left hip_x', 'Left Knee_x'],['Left Shoulder_y', 'Left hip_y', 'Left Knee_y']]}
        
        for lists in self.inputColumns[AngleCalc]:
            
            # Get the columns that is required for Angle Calc
            XTrainVariables, YTrainVariables, XTestVariables, YTestVariables = self.SeperateXYVariables(lists)
            assert len(XTrainVariables) == len(YTrainVariables)
            assert len(XTestVariables) == len(YTestVariables)
            self.XFeatures = lists

            # Check all the elements in YTestVaraible == -1
            assert all(x == -1 for x in YTestVariables) == True

            # Get only the column that has high correlation with Target variable
            X, Y, X_test, ColumnNameX, ColumnNameY, corrcoef = self.FindCorrelation(XTrainVariables, YTrainVariables, XTestVariables)

            # Perform Regression to get the missing values
            if corrcoef > 0.7:
                missing_values = self.PerformLinearRegression(X, Y, X_test)
            else:
                missing_values = self.PerformRandomForest(XTrainVariables, YTrainVariables, XTestVariables)

            # update the missing values in keypoint dict
            self.UpdateMissing(X_test, missing_values, ColumnNameX, ColumnNameY)

        self.CalculateAngles()
        self.drawPose(image_dir,image_output_path, gt_file)

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
            Knee_text = "Knee bend : " + str(self.left_keypoints[Frame]['Knee_bend'])[:8] + " degs"
            txt_size = cv2.getTextSize(Knee_text, cv2.FONT_HERSHEY_SIMPLEX, scale, cv2.FILLED)
            end_x = 20 + txt_size[0][0] 
            end_y = 20 - txt_size[0][1] 
            image = cv2.rectangle(image, (20,20), ((end_x+200), (end_y+200)), (255, 255, 255), cv2.FILLED)

            image = cv2.putText(image, Knee_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, scale*1.2, (0,0,255), 2)
            Knee_gt = "Knee bend gt : " + str(self.KneeGT[Frame])[:8] + " degs"
            image = cv2.putText(image, Knee_gt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, scale*1.2, (255,0,0), 2)

            # Hip bend values display
            Hip_text = "Hip bend : " + str(self.left_keypoints[Frame]['Hip_bend'])[:8] + " degs"
            image = cv2.putText(image, Hip_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, scale*1.2, (0,0,255), 2)
            Hip_gt = "Hip bend gt : " + str(self.HipGT[Frame])[:8] + "degs"
            image = cv2.putText(image, Hip_gt, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, scale*1.2, (255,0,0), 2)

            Frame = "Frame number : " + str(Frame)
            image = cv2.putText(image, str(Frame), (20, 170), cv2.FONT_HERSHEY_SIMPLEX, scale*1.2, (0,255,0), 2)
            filename = image_output_path +  '/' + filename
            cv2.imwrite(filename, image)


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
                

    def SeperateXYVariables(self, lists):
        """
        Function to get only left foot, left hip, left knee from the list and remove the missing 
        values

        Returns : X_array [left_hip, left_knee]
                  y_array [ left_foot]
        """
        X_train_array = []
        Y_train_array = []
        X_test_array = []
        Y_test_array = []

        
        for frame,values in (self.left_keypoints.items()):
            if float(values[lists[0]]) != -1.0 and float(values[lists[1]]) != -1.0 and float(values[lists[2]]) != -1.0: 
                
                # remove all negative keypoints and make them zero
                if float(values[lists[0]]) < 0 or float(values[lists[1]]) < 0 or float(values[lists[2]]) < 0: 
                        values[lists[0]] = 0.5 if (float(values[lists[0]]) < 0) else values[lists[0]]
                        values[lists[1]] = 0.5 if (float(values[lists[1]]) < 0) else values[lists[1]]
                        values[lists[2]] = 0.5 if (float(values[lists[2]]) < 0) else values[lists[2]]                
                X_train_array.append([values[lists[0]],values[lists[1]]])
                #X_train_array.append(values[4])
                Y_train_array.append(values[lists[2]])
            elif float(values[lists[2]]) == -1.0:
                X_test_array.append([values[lists[0]],values[lists[1]]])
                #X_test_array.append(values[4])
                Y_test_array.append(values[lists[2]])

        X_train_array = np.asarray((X_train_array),dtype = np.float)
        Y_train_array = np.asarray(Y_train_array,dtype = np.float)
        X_test_array = np.asarray((X_test_array),dtype = np.float)
        Y_test_array = np.asarray(Y_test_array,dtype = np.float)
        return X_train_array, Y_train_array, X_test_array, Y_test_array
        
    def PerformLinearRegression(self, XTrainVariables, YTrainVariables, XTestVariables):
        """
        Function to perform linear regression if XVaraibles are highly correlated 
        """
        # Normalize the variables
        XTrainVariables = XTrainVariables.reshape(-1,1)
        YTrainVariables = YTrainVariables.reshape(-1,1)
        Normalized_X, Normalized_Y = self.Normalize(XTrainVariables, YTrainVariables)
        
        # Choose X variable depending upon the correlation matrix
        X_train, X_val, y_train, y_val = train_test_split(Normalized_X, Normalized_Y, test_size=0.2, random_state=SEED)
        X_train = X_train.reshape(-1,1)
        X_val = X_val.reshape(-1,1)
        
        # Instantiate the Linear regressor
        lr = LinearRegression(fit_intercept = True)
                
        # Fit 'lr' to the training set
        lr.fit(X_train, y_train)
        # Predict the test set labels 'y_pred'
        y_pred = lr.predict(X_val)

        # Test it on the train dataset too
        y_train_hat = lr.predict(X_train)

        # Evaluate the train set RMSE
        rmse_train = MSE(y_train, y_train_hat)**(1/2)
        
        # Evaluate the val set RMSE
        rmse_val = MSE(y_val, y_pred)**(1/2)
        
        # Print the test set RMSE
        print('Train set RMSE of Linear regressor: {} '.format(rmse_train))
        print('Validation set RMSE of Linear regressor: {} '.format(rmse_val))

        # Predict the values of the Test data set
        y_test = lr.predict(XTestVariables.reshape(-1,1))
        
        return y_test
    
    def PerformRandomForest(self, XTrainVariables, YTrainVariables, XTestVariables):

        """
        Function to perform Random Forest Regression
        """
        
        # Choose X variable depending upon the correlation matrix
        YTrainVariables = YTrainVariables.reshape(-1,1)
        X_train, X_val, y_train, y_val = train_test_split(XTrainVariables, YTrainVariables, test_size=0.2, random_state=SEED)
        
        # Hyper parameter tuning for Random Forest regressor gives best hyper parameters
        #params = self.hyper_parameter_tuning(X_train, y_train, XTestVariables)
        #X_train = X_train.reshape(-1,1)
        #X_val = X_val.reshape(-1,1)

        # Instantiate a random forests regressor
        rf = RandomForestRegressor(random_state=SEED, n_estimators = 10, min_samples_leaf = 4)
        
        # Run this if hyper parameter tuning is done
        '''rf = RandomForestRegressor(bootstrap = params['bootstrap'], random_state=SEED, n_estimators=params['n_estimators'], 
                                min_samples_leaf = params['min_samples_leaf'] , max_features= params['max_features'],
                                max_depth= params['max_depth'], min_samples_split=params['min_samples_split'])'''
        
        # Fit 'rf' to the training set
        rf.fit(X_train, y_train.ravel())
        
        # Predict the test set labels 'y_pred'
        y_pred = rf.predict(X_val)

        # Test it on the train dataset too
        y_train_hat = rf.predict(X_train)

        # Evaluate the train set RMSE
        rmse_train = MSE(y_train, y_train_hat)**(1/2)
        
        # Evaluate the val set RMSE
        rmse_val = MSE(y_val, y_pred)**(1/2)
        
        # Print the test set RMSE
        print('Train set RMSE of Random Forest regressor: {} '.format(rmse_train))
        print('Validation set RMSE of Random Forest regressor: {} '.format(rmse_val))

        # Predict the values of the Test data set
        y_test = rf.predict(XTestVariables)
        
        return y_test
   
    def hyper_parameter_tuning(self, X_train, y_train, XTestVariables):
        
        # Number of trees in random forest #Creates this array : [100, 200, 300, 400, 500]
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)] 
        
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']

        # Maximum number of levels in tree : Creates this array : [5, 10, 15, 20, 25]
        max_depth = [int(x) for x in np.linspace(5, 25, num = 5)]
        max_depth.append(None) #Add none to the array

        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]

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
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=SEED, n_jobs = -1)

        # Fit the random search model
        rf_random.fit(X_train, y_train.ravel())

        # Predict the values of the Test data set
        y_test = rf_random.predict(XTestVariables)

        return rf_random.best_params_

    def FindCorrelation(self, XTrainVariables, YTrainVariables, XTestVariables):
        """
        Function to find the correlation matrix between the variables
        """
        
        corr0 = np.corrcoef(XTrainVariables[:,0], YTrainVariables)
        corr1 = np.corrcoef(XTrainVariables[:,1], YTrainVariables)

        print("Correlation of hip with foot {}".format(corr0))
        print("Correlation of knee with foot {}".format(corr1))

        if (corr0[1][0] > corr1[1][0]):
            return XTrainVariables[:,0], YTrainVariables, XTestVariables[:,0] , self.XFeatures[0] , self.XFeatures[2] , corr0[1][0]
        else:
            return XTrainVariables[:,1], YTrainVariables, XTestVariables[:,1] , self.XFeatures[1], self.XFeatures[2] ,corr0[0][1]

    def Normalize(self, XTrainVariables, YTrainVariables):

        """
        Function to normalize the input and output variables to be in the range [0,1]
        Returns : Normalized_x [array] : same size as XtrainVariables
                  Normalized_y [array] : same size as YTrainVaraibles
        """
        MinX = min(XTrainVariables)
        MaxX = max(XTrainVariables)
        Normalized_X = np.zeros(len(XTrainVariables),dtype =np.float)
        Normalized_Y = np.zeros(len(YTrainVariables),dtype =np.float)
        for index,x in enumerate(XTrainVariables):
            Normalized_X[index] = ((x - MinX)/(MaxX - MinX))
        
        MinY = min(YTrainVariables)
        MaxY = max(YTrainVariables)
        for index,x in enumerate(YTrainVariables):
            Normalized_Y[index] = ((x - MinY)/(MaxY - MinY))

        return Normalized_X, Normalized_Y

    def UpdateMissing(self, X_test, missing_values, ColumnNameX, ColumnNameY):
        
        """
        Update the missing values (-1) with regressed value in the keypoints dict
        """
        count = 0
        for i,values in enumerate(X_test):
            for index in sorted(self.left_keypoints):
                if (float(self.left_keypoints[index][ColumnNameX]) ==values) and (float(self.left_keypoints[index][ColumnNameY]) == -1):
                    self.left_keypoints[index][ColumnNameY] = str(missing_values[i])
                    count += 1
                    break
        print("Total missing values calculated : {}".format(count))

    def CalculateAngles(self):
        """
        Calculate the missing angles for Knee bend and Hip bend from the keypoint dict 
        and update the values in the keypoint dict with key Knee_bend, Hip_bend
        """
        
        # calculates Angles for Knee bend
        for frame,values in self.left_keypoints.items():
            A = math.sqrt( ((float(self.left_keypoints[frame]['Left hip_x']) - float(self.left_keypoints[frame]['Left Knee_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left hip_y']) - float(self.left_keypoints[frame]['Left Knee_y']))**2) ) 
            B = math.sqrt( ((float(self.left_keypoints[frame]['Left Knee_x']) - float(self.left_keypoints[frame]['Left foot_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left Knee_y']) - float(self.left_keypoints[frame]['Left foot_y']))**2) ) 
            C = math.sqrt( ((float(self.left_keypoints[frame]['Left hip_x']) - float(self.left_keypoints[frame]['Left foot_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left hip_y']) - float(self.left_keypoints[frame]['Left foot_y']))**2) ) 
            self.left_keypoints[frame]['Knee_bend'] = 180-math.degrees(math.acos( ( (C**2)-(A**2)-(B**2) )/(-2*A*B) ) )

            # calculate Angle for hip bend
            D = math.sqrt( ((float(self.left_keypoints[frame]['Left Shoulder_x']) - float(self.left_keypoints[frame]['Left hip_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left Shoulder_y']) - float(self.left_keypoints[frame]['Left hip_y']))**2) ) 
            E = math.sqrt( ((float(self.left_keypoints[frame]['Left hip_x']) - float(self.left_keypoints[frame]['Left Knee_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left hip_y']) - float(self.left_keypoints[frame]['Left Knee_y']))**2) ) 
            F = math.sqrt( ((float(self.left_keypoints[frame]['Left Shoulder_x']) - float(self.left_keypoints[frame]['Left Knee_x']))**2) + 
                            ((float(self.left_keypoints[frame]['Left Shoulder_y']) - float(self.left_keypoints[frame]['Left Knee_y']))**2) ) 
            
            self.left_keypoints[frame]['Hip_bend'] = 180-math.degrees(math.acos( ( (F**2)-(D**2)-(E**2) )/(-2*D*E) ) )

def get_args_parser():
    # Arguments PArser function
    parser = argparse.ArgumentParser('csv_keypoint ', add_help=False)
    parser.add_argument('--input', default=" ", type=str)
    parser.add_argument('--output', default=" ", type=str)
    parser.add_argument('--angleCalculation', default=1, type=int, 
                        help='Use 1 - for hip bend angle and 2 - knee bend angle')
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
                                    AngleCalc = args.angleCalculation, image_dir= args.image_dir,
                                    image_output_path=args.outputimagedir, gt_file = args.gt_file)
