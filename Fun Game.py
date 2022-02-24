import cv2 as c
import random
import time
import numpy as np
import mediapipe as mp
#For using the default webcam the key is 0, for other webcams its 1
Webcam = 0
capture = c.VideoCapture(Webcam)

#Fetching the frontal face dectection trained model with .xml file extension
FaceDetect=c.CascadeClassifier("D:\SaiPython\haarcascade_frontalface_default.xml")

#List of pets ❤️
string=['CAT','DOG','RABBIT','HAMSTER','GOLDFISH','PLATYPUS','PARROT','TURTLE','CHICK','<3 BIRDS','CUBS','PUPPY','BEAR','PEGION','DOVE','KITTEN','DUCK','MONKEY']

Mhands = mp.solutions.hands
hands=Mhands.Hands()
MDraw = mp.solutions.drawing_utils

while True:
    
    # Capture frame-by-frame
    ret, img = capture.read()
    imgGray = c.cvtColor(img,c.COLOR_BGR2GRAY)
    face = FaceDetect.detectMultiScale(imgGray,1.2,5)
    for (x,y,w,h) in face:
        c.rectangle(img,(x-10,y-40),(x+w+40,y+h-100),(150,50,100),-1)
        c.putText(img,random.choice(string),(x+5,y-5),c.FONT_ITALIC,1,(100,150,10),2)
            
    imgRGB = c.cvtColor(img,c.COLOR_BGR2RGB)
    result=hands.process(imgRGB)

    c.putText(img,"My Fav Pet <3",(190,30),c.FONT_ITALIC,1,(100,150,50),3)

    #Show your hand to stop 
    if result.multi_hand_landmarks:
        for handMark in result.multi_hand_landmarks:
                
            for F_Num,Mrk in enumerate (handMark.landmark):
                l,b,f=img.shape;
                Hx,Hy=int(Mrk.x*b),int(Mrk.y*l)
                #print(F_Num,Hx,Hy)
                
                if F_Num == 4:
                    p=Hx
                    q=Hy
                    c.circle(img,(Hx,Hy),20,(100,150,50),c.FILLED);
                if F_Num == 8:
                    c.circle(img,(Hx,Hy),20,(100,150,50),c.FILLED)
                    c.line(img,(p,q),(Hx,Hy),(100,150,50),3)
                    distance=((((Hx - p )**2) + ((Hy-q)**2) )**0.5)
                    #c.putText(img,str(int(distance)),(10,30),c.FONT_ITALIC,1,(0,250,0),1)
            #MDraw.draw_landmarks(img,handMark,Mhands.HAND_CONNECTIONS)

            while(distance>=50):
                c.putText(img,random.choice(string),(x+5,y-5),c.FONT_ITALIC,1,(100,150,10),2)
                time.sleep(5)
                exit()
                
    
    c.imshow('Game with Python', img)
    c.waitKey(1)
    
capture.release()
c.destroyAllWindows()
