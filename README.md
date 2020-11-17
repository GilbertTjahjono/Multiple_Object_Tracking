# Multiple_Object_Tracking
A multiple object tracking system built for Intel RealSense D435.

This project was built on Ultralytics' YOLOv3 Repository: https://github.com/ultralytics/yolov3

Getting Started:
1. Connect an Intel RealSense D435 to your laptop
2. Open command prompt
3. Change the directory to this repository
4. Run this following line: python tracking_v3.py --cfg cfg/yolov4.cfg --weights weights/yolov4.weights --names data/coco.names --half --n-object 2 --match-model matching_model/NN_GA.json

wait a bit and you should see this kind of window popping out on your screen

![image](https://github.com/GilbertTjahjono/Multiple_Object_Tracking/blob/main/data/ReadMe.png?raw=true)

5. In order to run the live plot mode, open a second command prompt
6. Run this following line: python Live_Plot.py

You can see my demo video here: https://bit.ly/project_videos (Multiple Object Tracking.mp4)
