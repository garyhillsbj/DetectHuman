#python3.11.5
import cv2
import imutils
# from detect_HOG import VideoCamera
from detect_yolo3 import VideoCamera
from flask import Flask, render_template, Response


# initialize the flask app
app = Flask(__name__)


# initialize the hog descriptor and set SVM to pre-trained pedestrian detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# route the app to the home page
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

def gen_frame(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def detect_object(object):
    return Response(gen_frame(VideoCamera(object)),mimetype='multipart/x-mixed-replace; boundary=frame')


# route the app to the image detection page
@app.route("/image")
def image():
    image=VideoCamera(None).get_image("images/image.jpg")
    cv2.imwrite('static/out.jpg',image)
    return render_template("image.html",user_image='static/out.jpg')

# route the app to the video detection page
@app.route('/show_video')
def show_video():
    return detect_object("images/video.mp4")
        
# route the app to the webcam detection page
@app.route("/show_webcam")
def show_webcam():
    return detect_object(0)
    
# run the app
if __name__ == '__main__':
    app.run(debug=True)