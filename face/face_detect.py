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
    
    def crop(self,image,box):
        height,width = image.shape[:2]
        print(image.shape)
        x0,y0,x1,y1 = box

        return image[max(0,y0):min(y1,height),max(0,x0):min(width,x1)]

        # def _isoutside(v,w):
        #     if v < 0 or v > w:
        #         return True
        #     return False

        # if not _isoutside(x0,width) or _isoutside(x1,width) or _isoutside(y0,height) or _isoutside(y1,height):
        # #     return None
        # # print([x0,y0,x1,y1])
        #     return image[y0:y1,x0:x1]

    def crop4restore(self,image,box,scale = 1.1):
        heigh,width = image.shape[:2]
        x0,y0,x1,y1 = box

        def _isoutside(v,w):
            if v < 0 or v > w:
                return True
            return False

        if _isoutside(x0,width) or _isoutside(x1,width) or _isoutside(y0,height) or _isoutside(y1,height):
            return None
        return image[y0:y1,x0:x1]


        


