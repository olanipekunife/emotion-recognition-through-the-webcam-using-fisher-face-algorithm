import cv2
import glob as gb
face_detector1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_detector2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_detector3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_detector4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
emotion_list = ["neutral", "anger", "disgust", "happy", "surprise"]


def faceDetection(emotion):
    # Get list of all images with emotion
    files = gb.glob("datasett\\%s\\*" % emotion)
    filenumber = 0
    for f in files:
        frame = cv2.imread(f)  # Open image
     

        try:
            # Resize face so all images have same size
            out = cv2.resize(frame, (100, 100))
            cv2.imwrite("dataset\\%s\\custom%s.jpg" %
                        (emotion, filenumber), out)  # Write image
        except:
            pass  # pass the file on error
        filenumber += 1  # Increment image number


if __name__ == '__main__':
    for emotion in emotion_list:
        faceDetection(emotion)  # Call our face detection module
