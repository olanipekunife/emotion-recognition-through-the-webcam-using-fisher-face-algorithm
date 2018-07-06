

# Import OpenCV2 for image processing
import cv2
from scipy import stats
# Import numpy for matrices calculations
import numpy as np
import signal
import sys
import argparse # Command-line parsing library
import nolearn.dbn as dbn
parser = argparse.ArgumentParser(description=(
    "Live emotion recognition from the webcam"))
parser.add_argument("--netFile",type=str, default="srainer",
                    help=("read file from which to read the network for testing the camera stream."))


args = parser.parse_args()
SMALL_SIZE = (100, 100)
# Parse the user given arguments
net = args.netFile

#emojis = ["neutral", "anger", "disgust", "happy"]
#emojis = ["neutral", "anger", "happy"] #nah
emojis = ["neutral", "happy"]#neutral
#emojis = ["neutral", "anger"] #na
#emojis = ["neutral", "surprise"]  # ns
models = []
WINDOW_NAME = 'Emotion Recognition'  
# Load prebuilt model for Frontal Face and eyes
# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX
# Initialize and start the video frame capture
def runemotiondetection():
    i = 0
    for i in range(6):
        cam = cv2.VideoCapture(0)
        if cam:
            print("found cam at ",i)
        # Loop
        while True:
            
            # Read the video frame
            ret, im = cam.read()

            # Convert the captured frame into grayscale
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            #Contrast Adjustment, histogram equalization
            gray = cv2.equalizeHist(gray)
            #remoe noise from image
           
           # gray = cv2.fastNlMeansDenoisingMulti(gray, 2, 5, None, 4, 7, 35)
            # Get all face from the video frame
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces) == 1:
            # For each face in faces
    #   Returns a list of length 4, with the two corners of the rectangle that define
    #   the position of the face: [x, y, w, h], where (x, y) and (w, h)
    #   are the defining corners of the rectangle.
                for(x, y, w, h) in faces:
                    print('face was detected')
                    # Create rectangle around the face
                    cv2.rectangle(im, (x - 20, y - 20),
                                (x + w + 20, y + h + 20), (0, 255, 0), 4)
                    # crop the image to the face
                    gray = gray[y:y + h, x:x + w]
                    # Get all face from the video frame
                    # eyes = eye_cascade.detectMultiScale(gray)
                    # for (ex,ey,ew,eh) in eyes:
                    #     cv2.rectangle(im, (ex, ey), (ex+ew,ey+eh), (0,255,0), 2)
                    gray = cv2.resize(gray, (100, 100))
                    #Equalize the image (needs to be done in the same way it has been with the training data)
                    #gray = equalizeFromFloatCLAHE(gray, SMALL_SIZE)
                    #Recognize the face belongs to which ID

                    # Load the trained mode
                    
                    # pred, conf = recognizer.predict(gray)
                    # print(pred, ' ', conf, ' ', emojis[pred])
                    emotion_guesses = np.zeros((len(models), 1))
                    for index in range(len(models)):
                        prediction, confidence = models[index].predict(gray)
                        print('prediction is', prediction,'confidence is ', confidence)
                        emotion_guesses[index][0] = prediction
                        # if confidence > 1000:
                        #     emotion_guesses[index][0] = prediction
                        #     #emotion_guesses[index][1] = confidence
                        # else:
                        #      emotion_guesses[index][0] = 0
                    print('emotion guesses ->',emotion_guesses)
                    pred = int(stats.mode(emotion_guesses)[0][0])
                    print(int(stats.mode(emotion_guesses)[0][0]), ' ', emojis[pred])

                    # Put text describe who is in the picture
                    cv2.rectangle(im, (x - 22, y - 90),
                                (x + w + 22, y - 22), (0, 255, 0), -1)
                    cv2.putText(im, emojis[pred], (x, y - 40), font, 2, (255, 255, 255), 3)
            else:
                if len(faces) != 0:
                    print(len(faces),' faces was detected')
            # Display the video frame with the bounded rectangle
            cv2.imshow(WINDOW_NAME, im)
            # If 'q' is pressed, close program
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            


        # Stop the camera
        cam.release()

        # Close all windows
        cv2.destroyAllWindows()

# exit the program if ctrl + c is pressed
def signal_handler(signal, frame):
    print ("The emotion recognition program will terminate.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def readnetwork():
    print('reading network ' + net)
    # for i in range(3):
    for i in range(3):#3 for neutral.xaml and 2 for others
        model_temp = cv2.face.FisherFaceRecognizer_create()
        model_temp.read('models/'+net+'100' + str(i) + '.xml')
        print("reading trainer", i)
        models.append(model_temp)


# def equalizeFromFloatCLAHE(x, reshapeSize=SMALL_SIZE):
#   x = x * 255
#   x = np.asarray(x, dtype='uint8')
#   y = x.reshape(reshapeSize)
#   y = equalizeCLAHE(y).reshape(-1)
#   return y / 255.0

# def equalizeCLAHE(x):
#       # Contrast Limited Adaptive Histogram Equalization
#   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
#   return clahe.apply(x)
if __name__ == '__main__':
    readnetwork()
    runemotiondetection()
