import cv2
import imutils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

class VideoCamera():
    def __init__(self,object):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        if object!=None:
            self.video = cv2.VideoCapture(object)
        
    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if success:
            image = imutils.resize(image, width=min(500, image.shape[1]))
            regions, _ = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
            for (x, y, w, h) in regions:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    
    def get_image(self,file):
        image = cv2.imread(file)
        image = imutils.resize(image, width=min(500, image.shape[1]))
        regions, _ = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return image