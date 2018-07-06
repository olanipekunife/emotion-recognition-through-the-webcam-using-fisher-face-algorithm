import cv2
import glob as gb
import random
import numpy as np
import matplotlib.pyplot as plt
import facedetection
import nolearn.dbn as dbn
#Emotion list
#emojis = ["neutral", "anger", "disgust", "happy"]
#emojis = ["neutral", "anger", "happy"]
#emojis = ["neutral", "happy"]#neutral100.xml
#emojis = ["neutral", "anger"] # na100.xml
#emojis = ["neutral", "surprise"]  # ns100.xml
emojis = ["anger"]
#emojis = ["happy"]
# emojis = ["neutral", "anger", "contempt", "disgust",
#           "fear", "happy", "sadness", "surprise"]
#Initialize  face classifier

facee = cv2.face.FisherFaceRecognizer_create()
data = {}
#Function defination to get file list, randomly shuffle it and split 67/33


def getFiles(emotion):
    files = gb.glob("dataset\\%s\\*" % emotion)
    random.shuffle(files)
    training = files[:int(len(files) * 0.90)]  # get first 90% of file list
    prediction = files[-int(len(files) * 0.10):]  # get last 10% of file list
    return training, prediction


def makeTrainingAndValidationSet():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emojis:
        training, prediction = getFiles(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            # convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # append image array to training data list
            training_data.append(gray)
            training_labels.append(emojis.index(emotion))  
        for item in prediction:  # repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emojis.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def runClassifier(i=0):
    training_data, training_labels, prediction_data, prediction_labels = makeTrainingAndValidationSet()

    print("training classifier suing the training data")
    print("size of training set is:", len(training_labels), "images")
    facee.train(training_data, np.asarray(training_labels))
    #facee.write('models/trainer.xml')
    facee.write('models/nsyuyu100' + str(i) + '.xml')
    
    print("classification prediction")
    counter = 0
    right = 0
    wrong = 0
    for image in prediction_data:
        pred, conf = facee.predict(image)
        if pred == prediction_labels[counter]:
            print(pred, ' ', emojis[pred])
            right += 1
            counter += 1
        else:
            wrong += 1
            counter += 1
    return ((100 * wrong) / (right + wrong))


#Now run the classifier
#bins = [0,10,20,30,40,50,60,70,80,90,100]

metascore = []
for i in range(0, 6):
    right = runClassifier(i)
    print("got", right, "percent wrong! for ->", i)
    metascore.append(right)
tit = ",".join(str("%.2f" % round(x, 2)) + "%" for x in metascore)
ids = [int(x) + 1 for x in range(len(metascore))]

all = np.mean(metascore)
okay = 'Error Probability of Trained Network - ' + str(tit) + ' Mean Score - ' + str("%.2f" % round(all, 2)) + "%"
#print("\n\nend score:", np.mean(metascore), "percent right!")
plt.bar(ids, metascore, label='Score', color='r')
plt.plot(ids, metascore, label='Score', color='b')
#plt.hist(metascore, bins, histtype='bar', color='r', rwidth=0.8, label="Score")
plt.ylabel('Range (0-100%)')
plt.xlabel('ids (1-10)')
plt.title(okay)
plt.legend()
plt.show()
