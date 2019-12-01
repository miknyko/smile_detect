import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from imutils.video import VideoStream

from face import FaceDetector
from smile import SmileDetector

FPS_value = 0.5
vs = VideoStream(src=0)
face_detector = FaceDetector(.9)
net = SmileDetector()
cwd = os.path.dirname(__file__)
net.weights_load(os.path.join(cwd,'smile','minivgg_weights.h5'))
threshold = .5

host = 'http://10.13.7.97:8000/'


def find_anchors():
    while True:
        frame = vs.read()
        faces = face_detector.detect(frame)
        key = input('{} faces detected, press `q` to quit finding anchors, otherwise press any key to start over.'.format(len(faces)))
        if key == 'q' or key == 'Q':
            break
    # upload()
    print("\n[info] contructing anchor data to be uploaded...")
    upload_face = []
    for i, face in enumerate(faces):
        tlx, tly, brx, bry = face[1:].astype('int32')
        tlx, tly, brx, bry = list(map(int, [tlx, tly, brx, bry]))
        face_image = face_detector.crop(frame, [tlx, tly, brx, bry], padding=True)
        obj = {
            'uid': 'person{}'.format(i + 1),
            'x0': tlx, 'y0': tly,
            'x1': brx, 'y1': bry,
            'cover': face_image.tolist()
        }
        upload_face.append(obj)
    data = {
        'event_name': args.event,
        'faces': upload_face
    }
    print("\n[info] requesting anchor API...")
    endpoint = 'api/anchor/'
    url = os.path.join(host, endpoint)
    response = requests.post(url, json=data)
    print(response.content)
    return response.status_code


def upload_smiles():
    frame = vs.read()
    # frame = cv2.imread('face/test.jpg')
    t1 = datetime.now()
    faces = face_detector.detect(frame)
    t2 = datetime.now()
    print("\n[info] face detection takes {}, {} faces detected".format(t2 - t1, len(faces)))
    if not len(faces):
        return
    feed = []
    for i, face in enumerate(faces):
        tlx, tly, brx, bry = face[1:].astype('int32')
        face_image = face_detector.crop(frame, [tlx, tly, brx, bry])
        face_image = cv2.resize(face_image, (100, 100))
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        feed.append(gray)
    feed = np.expand_dims(np.array(feed), axis=3)
    t1 = datetime.now()
    scores = net.model.predict(feed).squeeze(axis=1)
    is_smile = (scores > threshold)
    t2 = datetime.now()
    facesarray = np.array(faces)
    smile_faces = facesarray[is_smile]
    print("[info] smile assessment takes {}, {} smiles found".format(t2 - t1, len(smile_faces)))
    smile_scores = scores[is_smile]
    assert len(smile_faces) == len(smile_scores)

    smile_list = []
    for i in range(len(smile_faces)):
        smile = smile_faces[i]
        x0, y0, x1, y1 = list(map(int, smile[1:]))
        obj = {
            'score': float(smile_scores[i]),
            'x0': x0, 'y0': y0,
            'x1': x1, 'y1': y1,
            'face_image': face_detector.crop(frame, [x0, y0, x1, y1], padding=True).tolist()
        }
        smile_list.append(obj)
    data = {
        'timestamp': str(datetime.now()),
        'event_name': args.event,
        'smiles': smile_list
    }
    endpoint = 'api/smiles/'
    url = os.path.join(host, endpoint)
    print("[info] requesting smiles API...")
    t1 = datetime.now()
    response = requests.post(url, json=data)
    t2 = datetime.now()
    print("[info] smiles API takes {}".format(t2 - t1))
    print(response.content)

    

    return faces,frame,scores


def main_loop():
    # init
    vs.start()
    # upload anchor
    t1 = datetime.now()
    find_anchors()
    t2 = datetime.now()
    print("\n[info] Anchor API takes {}".format(t2 - t1))

    tick = datetime.now()
    while True:
        tock = datetime.now()
        elapsed = (tock - tick).total_seconds()
        if elapsed < (1. / FPS_value):
            continue
        tick = datetime.now()
        faces,frame,scores = upload_smiles()
        if args.camerawindow:
            for i, face in enumerate(faces):    
                tlx, tly, brx, bry = face[1:].astype('int32')
                cv2.rectangle(frame, (tlx, tly), (brx, bry), color=(0, 0, 255), thickness=2)
                # cv2.putText(frame, res[i], (tlx + 10, bry + 10), cv2.FONT_HERSHEY_SIMPLEX, .5, color=(0, 0, 255))
                # cv2.putText(frame, "{:.2f}%".format(float(scores[i]) * 100), (tlx + 10, bry - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, color=(0, 0, 255))
            cv2.imshow('camera',frame)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--res',default = 1)
    parser.add_argument('-w','--camerawindow',default = False)
    parser.add_argument('-pi','--picamera',default = False)
    parser.add_argument('-e','--event',type=str)

    args = parser.parse_args()
    print("\n\n\n\n\n\n\n\n")
    main_loop()
 
    

            


            







