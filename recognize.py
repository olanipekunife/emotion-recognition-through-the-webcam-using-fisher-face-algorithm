import cv2
# Load prebuilt model for Frontal Face and eyes
# Create classifier from prebuilt model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Initialize and start the video frame capture
cap = cv2.VideoCapture(0)

while True:
      # Read the video frame
    ret,img = cap.read()
    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  

    cv2.imshow('Colored Image',img)
    cv2.imshow('gray Image', gray)

    # if the escape jey is press exit the program
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
