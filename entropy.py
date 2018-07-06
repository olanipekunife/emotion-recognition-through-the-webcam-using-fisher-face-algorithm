import cv2
import math
# Load prebuilt model for Frontal Face and eyes
# Create classifier from prebuilt model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Initialize and start the video frame capture
#cap = cv2.VideoCapture(0)
i = 0
for i in range(6):
    cap = cv2.VideoCapture(0)
    if cap:
        print("found cam at ", i)
        # Loop
    while True:
        # Read the video frame
        ret, img = cap.read()
        # Convert the captured frame into grayscale
        rgbhistogram = cv2.calcHist([img],[0],None,[256],[0,256])
        for rgb in range(3):
            totalpixels = sum(rgbhistogram[rgb * 256:(rgb + 1) * 256])
            ent = 0.0
            for col in range(rgb * 256, (rgb + 1) * 256):
                freq = float(rgbhistogram[col]) / totalpixels
                if freq > 0:
                    ent = ent + freq * math.log(freq, 2)
                ent = - ent
                print(ent)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Get all face from the video frame
        faces = face_cascade.detectMultiScale(gray)
        # get face coordinate
        for (x, y, w, h) in faces:
            # Create rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            #get eye coordinate
            for (ex, ey, ew, eh) in eyes:
                # Create rectangle around the eye
                cv2.rectangle(roi_gray, (ex, ey),
                            (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('img', img)
        # if the escape jey is press exit the program
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
