import cv2
import numpy as np
#For using the default webcam the key is 0, for other webcams its 1
Webcam = 0
capture = cv2.VideoCapture(Webcam)
#Fetching the frontal face dectection trained model with .xml file extension
FaceDetect=cv2.CascadeClassifier("D:\SaiPython\haarcascade_frontalface_default.xml")

while True:
    # Capture frame-by-frame
    ret, image = capture.read()
    imgGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face = FaceDetect.detectMultiScale(imgGray,1.2,5)
    for (x,y,w,h) in face:
        cv2.rectangle(image,(x,y),(x+w+15,y+h+15),(0,255,0),4)
        cv2.putText(image," Face Detected",(x-10,y-10),cv2.FONT_ITALIC,1,(0,255,0),2)
        
    cv2.imshow('image', image)
    cv2.waitKey(1)
    
capture.release()
cv2.destroyAllWindows()
