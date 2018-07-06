import cv2
import numpy as np
from skimage.transform import resize
# Create window for image display
CASCADE_FN = "haarcascade_frontalface_default.xml"

# The scale used for face recognition.
# It is important as the face recognition algorithm works better on small images
# Also helps with removing faces that are too far away
RESIZE_SCALE = 3
RECTANGE_COLOUR = (117, 30, 104)
BOX_COLOR = (255, 255, 255)
THICKNESS = 2
SMALL_SIZE = (40, 30)
SQUARE_SIZE = (48, 48)

def equalizeFromFloatCLAHE(x, reshapeSize=SMALL_SIZE):
  x = x * 255
  x = np.asarray(x, dtype='uint8')
  y = x.reshape(reshapeSize)
  y =  equalizeCLAHE(y).reshape(-1)
  return y / 255.0

def equalizeCLAHE(x):
      # Contrast Limited Adaptive Histogram Equalization
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
  return clahe.apply(x)

def preprocess(image, return_vector=False):
    """Preprocess the input image according to the face coordinates detected
    by a face recognition engine.

    This method:
     * crops the input image, keeping only the face given by faceCoordinates
     * transforms the picture into black and white
     * equalizes the input image

   If return_vector is True, returns a vector by concatenating the rows of the
   processed image. Otherwise, a matrix (2-d numpy array) is returned.

   This method needs to be called both for training and testing.
   """
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Step 1: crop the the image
    # cropped = cropFace(image, faceCoordinates)

    # Step 2: Resize
    resized = np.ascontiguousarray(resize(image, SMALL_SIZE))

    # Step 3: Equalize the image (needs to be done in the same way it has been with the training data)
    equalized = equalizeFromFloatCLAHE(resized, SMALL_SIZE)
    if return_vector:
        return equalized
    return np.reshape(equalized, SMALL_SIZE)

def scale(data):
      # return preprocessing.scale(data, axis=1)
  data = data / data.std(axis=1)[:, np.newaxis]
  data = data - data.mean(axis=1)[:, np.newaxis]

  # print data.std(axis=1).sum()
  # print np.ones((data.shape[0]), dtype='float')
  # assert np.array_equal(data.std(axis=1), np.ones((data.shape[0]), dtype='float'))
  # assert np.array_equal(data.mean(axis=1), np.zeros(data.shape[0]))
  return data

def cropFace(image, faceCoordinates):
      return image[faceCoordinates[1]: faceCoordinates[3],
                faceCoordinates[0]: faceCoordinates[2]]

def getFaceCoordinates(image):
  """Uses openCV to detect the face preent in the input image.

  Returns a list of length 4, with the two corners of the rectangle that define
  the position of the face: [x1, y1, x2, y2], where (x1, y1) and (x2, y2)
  are the defining corners of the rectangle.
  """

  cascade = cv2.CascadeClassifier(CASCADE_FN)
  img_copy = cv2.resize(image, (image.shape[1] / RESIZE_SCALE,
                                image.shape[0] / RESIZE_SCALE))
  gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
  gray = cv2.equalizeHist(gray)
  rects = cascade.detectMultiScale(gray, 1.2, 3)

  # If there is no face or if we have more than 2 faces return None
  # because we do not deal with that yet
  if len(rects) != 1:
    return None

  r = rects[0]
  corners = [r[0], r[1], r[0] + r[2], r[1] + r[3]]

  return list(map((lambda x: RESIZE_SCALE * x), corners))
