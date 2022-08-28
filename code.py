import cv2
import numpy as np
import face_recognition #module to detect face recognition
import os #to read the images os module is used
from datetime import datetime #to mark attendence we need date and time module


path = 'images' #the folder name from which we import the images
images = []
personNames = []
myList = os.listdir(path)
print(myList)
for cu_img in myList: #H	ere we read the images by calling cv2 module
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img) #then we will store the images in this list
    personNames.append(os.path.splitext(cu_img)[0]) #And we will store the person names in this list
print(personNames)


def faceEncodings(images): #to detect the n no of images and it will detect 128 unique things which is differ from the other person in your face
# alo in this hog algorithm works to find out encoding
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name): # to mark the attendence 
    with open('Attendance.csv', 'r+') as f: # and store it in that location with name date and time
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!') # whenever the encoding is complete it will print this..

cap = cv2.VideoCapture(0) # to read the camera video capture is used in cv2 here if we are using internal camera so here we use 0 else for webcam we are using 1

while True: # if this is true then camera opens in an infinite loop till we press enter..
    ret, frame = cap.read() # use to read camera
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0, 255, 0), 2) # it will create a rectangle of green colour
            cv2.rectangle(frame, (x1,y2 - 35),(x2,y2), (0, 255, 0), cv2.FILLED) # for name
            cv2.putText(frame, name, (x1 + 6,y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13: # it will chech if our ascii value is 13 means enter key or not..
        break

cap.release()
cv2.destroyAllWindows() 