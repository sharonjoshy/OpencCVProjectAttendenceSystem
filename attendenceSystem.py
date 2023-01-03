import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
path = 'data'

images = []

classnames = [] #for collecting name of image

mylist = os.listdir(path) #for collecting pixal values of image

# print(mylist)

for cl in mylist:
    classnames.append(os.path.splitext(cl)[0])
    images.append(cv2.imread(f'{path}/{cl}'))

# print(classnames)

def findencodings(images):
    encodelist=[]  # finding encodings  of known faces

    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return(encodelist)

encodelistknownfaces = findencodings(images) #findig encodings of known faces

print("ENCODING COMPLETED............")


def attendence(name):
    with open(r"C:\Users\sharo\Desktop\OpenCV\project_1\attendenceResult.csv","r+") as f:
        data_in_csv = f.readlines()
    
    
        namelist = []
        for i in data_in_csv:
            name_in_csv = i.split(',')
            namelist.append(name_in_csv[0])
        
        if name not in namelist:
            dtandtime = datetime.now()
            date = (dtandtime.strftime("%d-%m-%y"))
            time = (dtandtime.strftime("%I:%M %p"))
        
            cv2.putText(img,"marking completed successfully",(x1+2,y1-15),cv2.FONT_HERSHEY_COMPLEX,.5,(0,255,0),1)
            f.writelines(f'\n{name},{time},{date}')
        





# ================================================

# webcam

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imagesmall=cv2.resize(img,(0,0),None,0.25,0.25)  #resizing image

    faceinframe = face_recognition.face_locations(imagesmall)  # taking face from image
    encoded_face=face_recognition.face_encodings(imagesmall,faceinframe)  #find encodings from faces in camera


# we need to comapare image encodlist and encode_face  

    for encface,faceloc in zip(encoded_face,faceinframe):
        matches=face_recognition.compare_faces(encodelistknownfaces,encface)
        facedistance=face_recognition.face_distance(encodelistknownfaces,encface)
        
        

        #print(facedistance)  #less value will be our output

        matchindex = np.argmin(facedistance)      #to take minimum value
        
        if matches[matchindex]:
            name=classnames[matchindex].upper()
            y1,x2,y2,x1 = faceloc
            # we scaled down by 4 times
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4  # to scale up by 4 times
            cv2.rectangle(img,(x1,y1),(x2+5,y2+5),(0,255,0),3)
            cv2.rectangle(img,(x1,y2+3),(x2+5,y2+27),(0,255,0),-2)
            cv2.putText(img,name,(x1,y2+20),cv2.FONT_HERSHEY_COMPLEX,.6,(0,0,255),1)
            attendence(name)

            print(name)





    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break