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

To Fill out missing keypoints by regression run the following command

Code should be modified depending on the XVariables(Input Features) if they are highly correlated use Linear regression function, if not use Random Forest regressor
```
python3 keypt_regression.py --input KeypointsFromRevamp.csv --output Final.csv --angleCalculation 1 --gt_file GT.csv 
                            --image_dir "Path/To/Frames/Folder" --outputimagedir "Path/To/Output/FramesFolder"

```

* angleCalculation 
1 - if the keypoints of (hip,knee,foot) is missing 
2 - if the keypoints of (shoulder,knee,hip) is missing
* gt_file - GT csv containing knee bend and hip bend sensor angles 
* image_dir - path to the frame folder from dataset
* outputimagedir - path to the folder to store final images with keypoints and angles calculated on it
  
