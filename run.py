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
parser.add_argument('-v','--videostream',default = True)
parser.add_argument('-s','--stillimage',default = False)

args = parser.parse_args()

def norm_image(x):
    return x / 127.5 - 1

def video_main():
    cwd = os.getcwd()
    face_detector = FaceDetector(0.8) # scores below 0.8 would not be recognized as a face
    smile_detector = SmileDetector()
    smile_detector.weights_load(os.path.join(cwd,'smile','minivgg_weights.h5'))
    vs = VideoStream(src=0).start()
    time.sleep(1)
    fps = FPS().start()
    labels = ['calm','smile']
    smile_count = 0
    time_count =0

    while True:
        gray_faces = []
        frame = vs.read()
        faces = face_detector.detect(frame)  # using face detector to detect the face
        if len(faces):
            time_count += 1
        for i,face in enumerate(faces):
            conf = face[0]
            tlx,tly,brx,bry = face[1:].astype('int32')
            face_image = face_detector.crop(frame,[tlx,tly,brx,bry])
            face_image = cv2.resize(face_image,(100,100)) #  resize the cropped face image to 100*100, for the minivgg net inference input
            gray = cv2.cvtColor(face_image,cv2.COLOR_RGB2GRAY)
            gray_faces.append(gray)

        if len(gray_faces):
            batch_faces = np.array(gray_faces)
            batch_faces = np.expand_dims(gray_faces,axis=3)
            score = smile_detector.model.predict(norm_image(batch_faces)) # using smile detector to classify the face whether it's smiling
            pred = (score > 0.5).astype('uint8')  # scores above 0.5 would be considered as a smile
            res = [labels[int(p)] for p in pred]
            for i in res:
                if i == labels[1]:
                    smile_count += 1   # if a smile face is detected ,the smile count increased

            for i, face in enumerate(faces):

                tlx,tly,brx,bry = face[1:].astype('int32')
                cv2.rectangle(frame,(tlx,tly),(brx,bry),color = (255,0,0),thickness=3)
                cv2.putText(frame,res[i],(tlx + 10,tly + 10),cv2.FONT_HERSHEY_SIMPLEX,.5,color=(255,0,0))
                cv2.putText(frame,"{:.2f}%".format(float(score[i]) * 100),(tlx + 10,tly - 10),cv2.FONT_HERSHEY_SIMPLEX,.5,color=(255,0,0))

        fps.update()
        cv2.imshow('camera',frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    fps.stop()
    vs.stop()
    print(f'[INFO] YOUR HAPPY SCORE IS {smile_count/time_count:.2f}!')        

if __name__ == '__main__':
    video_main()
    
