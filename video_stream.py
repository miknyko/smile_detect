import os
import cv2
import tensorflow as tf
import time
import numpy as np
import argparse

from face import FaceDetector
from smile import SmileDetector
from imutils.video import VideoStream,FPS

parser = argparse.ArgumentParser()
parser.add_argument('-r','--res',default = (480,640))
parser.add_argument('-w','--camerawindow',default = False)
parser.add_argument('-pi','--picamera',default = False)
args = parser.parse_args()

if args.picamera:
    from imutils.video import pivideostream

def norm_image(x):
    return x / 127.5 - 1

if __name__ == '__main__':
    cwd = os.path.dirname(__file__)
    detector_1 = FaceDetector(0.8)
    vs = VideoStream(src=0,usePiCamera=args.picamera,resolution = args.res).start()
    time.sleep(1)
    fps = FPS().start()

    detector_2 = SmileDetector()
    detector_2.weights_load(os.path.join(cwd,'smile','minivgg_weights.h5'))

    labels = ['calm','smile']

    while True:
        tick = time.time()
        frame = vs.read()
        tock = time.time()
        # print(f'video caputre takes {tock - tick}')

        tick = time.time()
        faces = detector_1.detect(frame)
        tock = time.time()
        # print(f'face detection takes {tock - tick}')

        gray_faces = []
        tick = time.time()
        for i,face in enumerate(faces):
            confidence = face[0]
            tlx,tly,brx,bry = face[1:].astype('int32')
            print([tlx,tly,brx,bry])
            face_image = detector_1.crop4inf(frame,[tlx,tly,brx,bry])
            print(face_image.shape)
            face_image = cv2.resize(face_image,(100,100))
            gray = cv2.cvtColor(face_image,cv2.COLOR_RGB2GRAY)
            gray_faces.append(gray)

            
        tock = time.time()
        # print(f'face detection post-processing takes {tock - tick}')

        if len(gray_faces):
            batch_faces = np.array(gray_faces)
            batch_faces = np.expand_dims(gray_faces,axis=3)
            tick = time.time()
            y = detector_2.model.predict(norm_image(batch_faces))
            print(y)
            pred = (y > 0.5).astype('uint8')
            tock = time.time()
            res = [labels[int(p)] for p in pred]
            print(res)
            # print(f'face inference taks {tock - tick}')

            for i, face in enumerate(faces):
                
                tlx, tly, brx, bry = face[1:].astype('int32')
                cv2.rectangle(frame, (tlx, tly), (brx, bry), color=(0, 0, 255), thickness=2)
                cv2.putText(frame, res[i], (tlx + 10, bry + 10), cv2.FONT_HERSHEY_SIMPLEX, .5, color=(0, 0, 255))
                cv2.putText(frame, "{:.2f}%".format(float(y[i]) * 100), (tlx + 10, bry - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, color=(0, 0, 255))
        

        fps.update()
        if args.camerawindow:
            cv2.imshow('camera',frame)

        key = cv2.waitKey(1)
        if key == 'q':
            break

    fps.stop()
    vs.stop()
    

            


            







