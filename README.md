
Detect Human web-app coded in Python using OpenCV and Flask<br>
## Dependencies
download: yolov3-320.weights in https://pjreddie.com/media/files/yolov3.weights<br>
download: yolov3-320.cfg https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg<br>
download: https://github.com/pjreddie/darknet/blob/master/data/coco.names<br>
python3.11.5<br>
Flask==3.0.0<br>
opencv-python==4.8.1.78<br>
imutils==0.5.4<br>

## Guide
- Clone this repository.<br>
      git clone https://github.com/annopsamu/cs_game_bot
- Install requirements using:<br>
      pip install -r requirements.txt
- Run the main file:<br>
      python3 main.py
- Now you can use Human Detection in 3 different modes.

## Note
- You can use different images just replace the image in root dir and rename it to `image.jpg`
- Same way you can use different videos by replacing the video in root dir and renaming it to `video.mp4`

