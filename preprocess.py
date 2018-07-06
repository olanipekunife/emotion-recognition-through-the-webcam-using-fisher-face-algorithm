import cv2
import glob as gb
import facedetection
face_detector1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_detector2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_detector3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_detector4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
# emotion_list = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
emotion_list = ["anger"]

def facdetection(emotion):
    files = gb.glob("sorted_set\\%s\\*" %emotion) #Get list of all images with emotion
    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        print("Checking for face in: %s" % f)

        #Detect face using 4 different classifiers
        face1 = face_detector1.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = face_detector2.detectMultiScale(gray, scaleFactor=1.1,
         minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = face_detector3.detectMultiScale(gray, scaleFactor=1.1,
         minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = face_detector4.detectMultiScale(gray, scaleFactor=1.1,
         minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face1) == 1:
            facefeatures = face1
        elif len(face2) == 1:
            facefeatures == face2
        elif len(face3) == 1:
            facefeatures = face3
        elif len(face4) == 1:
            facefeatures = face4
        else:
            facefeatures = ""
        
        #Cut and save face
       
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print("face found in file: %s" %f)
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("final_dataset\\%s\\austom%s.png" %(emotion, filenumber), out) #Write image
            except:
               pass #pass the file on error
       
        filenumber += 1 #Increment image number
if __name__ == '__main__':
    for emotion in emotion_list: 
        facdetection(emotion) #Call our face detection module
