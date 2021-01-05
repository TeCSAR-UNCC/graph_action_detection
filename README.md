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
