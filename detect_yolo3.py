# TechVidvan Vehicle counting and Classification

# Import necessary packages

import cv2
import numpy as np

class VideoCamera():
    def __init__(self,object):
        if object!=None:
            self.cap = cv2.VideoCapture(object)
        self.input_size = 320
        # Detection confidence threshold
        self.confThreshold =0.2
        self.nmsThreshold =0.2

        # class index for our required detection classes
        self.required_class_index = [0]
        self.detected_classNames = []
        ## Model Files
        modelConfiguration = 'yolov3-320.cfg'
        modelWeigheights = 'yolov3-320.weights'
        # configure the network model
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)
        # Configure the network backend
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #----------gpu--------
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) #------------gpu---------

    # Function for finding the center of a rectangle
    def find_center(self, x, y, w, h):
        x1=int(w/2)
        y1=int(h/2)
        cx = x+x1
        cy=y+y1
        return cx, cy
        
    # Function for finding the detected objects from the network output
    def postProcess(self,outputs,img):
        height, width = img.shape[:2]
        boxes = []
        classIds = []
        confidence_scores = []
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if classId in self.required_class_index:
                    if confidence > self.confThreshold:
                        w,h = int(det[2]*width) , int(det[3]*height)
                        x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                        boxes.append([x,y,w,h])
                        classIds.append(classId)
                        confidence_scores.append(float(confidence))

        # Apply Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, self.confThreshold, self.nmsThreshold)
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 1)

    def get_frame(self):
        success, img = self.cap.read()
        if success==True:
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.input_size, self.input_size), [0, 0, 0], 1, crop=False)
            # Set the input of the network
            self.net.setInput(blob)
            layersNames = self.net.getLayerNames()            
            outputNames = [(layersNames[i - 1]) for i in list(self.net.getUnconnectedOutLayers())]        
            # Feed data to the network
            outputs = self.net.forward(outputNames)        
            # Find the objects from the network output
            self.postProcess(outputs,img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def get_image(self, file):
        img = cv2.imread(file)
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.input_size, self.input_size), [0, 0, 0], 1, crop=False)
        # Set the input of the network
        self.net.setInput(blob)
        layersNames = self.net.getLayerNames()            
        outputNames = [(layersNames[i - 1]) for i in list(self.net.getUnconnectedOutLayers())]        
        # Feed data to the network
        outputs = self.net.forward(outputNames)        
        # Find the objects from the network output
        self.postProcess(outputs,img)
        return img
