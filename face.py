import cv2
import numpy as np

def nothing(x):
    pass


face_cascade_path = '/Users/Ohkado/Documents/Python/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml'
eye_cascade_path ='/Users/Ohkado/Documents/Python/opencv-master/data/haarcascades/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

cv2.namedWindow('video image')
cv2.namedWindow('face image')
cv2.createTrackbar('ratio','video image', 25, 50, nothing)

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    ratio = cv2.getTrackbarPos('ratio', 'video image')
    face =[''] * 10
    composition = np.zeros(shape=(60,60,3))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3,minSize = (60,60))
    for i in range(0,len(faces)):
        x = faces[i][0]
        y = faces[i][1]
        w = faces[i][2]
        h = faces[i][3]
        #for x, y, w, h in faces:
            #small = cv2.resize(img[y: y + h, x: x + w], None, fx=(ratio+3)/100, fy=(ratio+3)/100, interpolation=cv2.INTER_NEAREST)
            #face = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        face[i] = cv2.resize(img[y: y+h, x: x+w], (60,60))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        composition += face[i]
        #face = img[y: y + h, x: x + w]
        #face_gray = gray[y: y + h, x: x + w]
        #eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5)
        #for (ex, ey, ew, eh) in eyes:
            #small = cv2.resize(face[ey: ey + eh, ex: ex + ew], None, fx=(ratio+3)/100, fy=(ratio+3)/100, interpolation=cv2.INTER_NEAREST)
            #img[ey: ey + eh, ex: ex + ew] = cv2.resize(small, (ew, eh), interpolation=cv2.INTER_NEAREST)

            #cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            #face[ey: ey + eh, ex: ex + ew] = [0, 0, 0]
    composition /= 255 * len(faces)
    composition = cv2.resize(composition,(200,200))
    cv2.imshow('face image', composition)
    cv2.imshow('video image', img)
    key = cv2.waitKey(10)
    if key == 27:  # ESCキーで終了
        break
cap.release()
cv2.destroyAllWindows()
