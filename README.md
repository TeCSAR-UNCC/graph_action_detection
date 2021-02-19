# graph_action_detection
graph based action detection and mobility assessment.

readme creation is in progress.

To extract video frames:
```
python3 extract.py --input input_video_path  --output output_folder_path
```
To average the angles in a basic csv to the same length:
```
python3 avg.py --input input_video_path  --output output_folder_path --avg angles_list_shrink
```

To execute the main code:
```
python3 keypoint-MA.py
```

## Missing Keypoint imputation using Random Forest regressor

CSV with missing keypoints sample is provided [here](https://github.com/TeCSAR-UNCC/graph_action_detection/blob/main/KeypointsFromRevamp.csv). 
The script expects the keypoints in this format with these column names.

EfficientHRnet output missing keypoints as -1.0. The angle calculation is done using 3 point method which requires all the three keypoints to calculate angles. For Example:
* Knee bend (angle) - Calculated between Hip, Knee and Foot keypoint. 
* Hip bend (angle) - Calculated between Shoulder, Hip and Knee Keypoint.
* Angles calculation is done using [Law of cosines](https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points).
Since the keypoints are missing the resulting angles are -1.0 which makes it difficult to compare to the wearable sensors which in turn is considered as gt in our experiment.

#### Missing Keypoints can be filled Offline using Random forest by following the below steps
  * Write the keypoints from the keypoint detector into CSV like [this](https://github.com/TeCSAR-UNCC/graph_action_detection/blob/main/KeypointsFromRevamp.csv) for offline regression.
  * Take only the columns with missing values and find its highly correlated columns For example : If the keypoints are missing in Foot then Foot x,y coordinates is highly correlated to knee than hip or shoulder.
  * Take the X Variable(Ex : Knee_keypoint x,y) and Y Variable (Ex : Foot_keypoint x,y) and seperate it to train and test dataset. The test dataset has the datapoints with -1.0 values. Then perform train,val split on train dataset.
  * Perform Random regression to get multi-ouput and then perform angle calculation on top of it.
  * Script can be found [here](https://github.com/TeCSAR-UNCC/graph_action_detection/blob/main/multi_output_reg.py)

## Pre-Requisite
```
pip install numpy
pip install opencv-python
pip install -U scikit-learn
pip install filterpy
```

## Steps to run
```
python3 multi_output_reg.py --input /path/To/KeypointsFromDetector.csv
                --output /Path/To/RandomForestFinal.csv                
                --gt_file /Path/To/GT.csv
                --image_dir /Path/To/imagesondataset
                --outputimagedir /Path/To/RandomForest_FinalImages
                --X1 Left Knee_x
                --X2 Left Knee_y
                --Y1 Left foot_x
                --Y2 Left foot_y
 ```
 Where 
 * --input - Csv file with Missing keypoint detector
 * --output - Path to Csv file to write the results
 * --gt_file - Optional (if you want to display groundtruth on the images)
 * --image_dir - Optional ( Location of the input image dataset)
 * --outputimagedir - Optional ( Location to store the images with keypoints after regression for visualization)
 * --X1 - Dependent variable column Left Knee_x
 * --X2 - Dependent variable column Left Knee_y
 * --Y1 - Target variable column Left foot_x
 * --Y2 - Target variable column Left foot_y
 

  
