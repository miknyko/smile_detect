import cv2
import tensorflow as tf
import numpy as np
import os

cwd = os.path.dirname(__file__)

class _FaceDetector():
    PrototextPath = os.path.join(cwd,'deploy.prototxt.txt')
    FDModelPath = os.path.join(cwd,'res10_300x300.caffemodel')

    def __init__(self,threshold = 0.5):
        self.threshold = threshold
        self.model = cv2.dnn.readNetFromCaffe(self.PrototextPath,self.FDModelPath)

    def detect(self,image):
        height,width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
        self.model.setInput(blob)
        detections = self.model.forward()
        boxes = []

        for i in range(0,detections.shape[2]):
            confidence_value = detections[0,0,i,2]
            if confidence_value < self.threshold:
                continue
            box = detections[0,0,i,3:7]
            if (box > 1).sum():
                continue
            box = box * np.array([width,height,width,height])
            pred = np.concatenate([[confidence_value],box],axis = 0)
            boxes.append(pred)
        return boxes
    
    def crop(self,image,box,padding = False):
        height,width = image.shape[:2]
        print(image.shape)
        x0,y0,x1,y1 = box

        if padding:
            height,width = image.shape[:2]
            x0,y0,x1,y1 = box
            longer = max((x1 - x0),(y1 - y0))
            xc = int((x0 + x1) / 2)
            yc = int((y0 + y1) / 2)
            longer_half = int(longer/2)
            x0 = xc - longer_half
            x1 = xc + longer_half
            y0 = yc - longer_half
            y1 = yc + longer_half


        return image[max(0,y0):min(y1,height),max(0,x0):min(width,x1)]


        


