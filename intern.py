import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM as lstm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from keras import regularizers

import os

mylist= os.listdir('C:/Users/Nouman/Desktop/Emotion-Analysis-Using-Speech-master/Actor_10/')

##Plotting the audio file's waveform and its spectrogram

data, sampling_rate = librosa.load('C:/Users/Nouman/Desktop/Emotion-Analysis-Using-Speech-master/Actor_10/03-02-01-01-01-01-09.wav')

##% pylab inline
import os
import pandas as pd
import librosa
import glob

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)


import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import sys


sr,x = scipy.io.wavfile.read('C:/Users/Nouman/Desktop/Emotion-Analysis-Using-Speech-master/Actor_10/03-02-01-01-01-01-09.wav')

## Parameters: 10ms step, 30ms window
nstep = int(sr * 0.01)
nwin  = int(sr * 0.03)
nfft = nwin

window = np.hamming(nwin)

## will take windows x[n1:n2].  generate
## and loop over n2 such that all frames
## fit within the waveform
nn = range(nwin, len(x), nstep)

X = np.zeros( (len(nn), nfft//2) )

for i,n in enumerate(nn):
    xseg = x[n-nwin:n]
    z = np.fft.fft(window * xseg, nfft)
    X[i,:] = np.log(np.abs(z[:nfft//2]))

plt.imshow(X.T, interpolation='nearest',
    origin='lower',
    aspect='auto')

plt.show()



##setting the labels

feeling_list=[]
for item in mylist:
    if item[6:-16]=='02' and int(item[18:-4])%2==0:
        feeling_list.append('female_calm')
    elif item[6:-16]=='02' and int(item[18:-4])%2==1:
        feeling_list.append('male_calm')
    elif item[6:-16]=='03' and int(item[18:-4])%2==0:
        feeling_list.append('female_happy')
    elif item[6:-16]=='03' and int(item[18:-4])%2==1:
        feeling_list.append('male_happy')
    elif item[6:-16]=='04' and int(item[18:-4])%2==0:
        feeling_list.append('female_sad')
    elif item[6:-16]=='04' and int(item[18:-4])%2==1:
        feeling_list.append('male_sad')
    elif item[6:-16]=='05' and int(item[18:-4])%2==0:
        feeling_list.append('female_angry')
    elif item[6:-16]=='05' and int(item[18:-4])%2==1:
        feeling_list.append('male_angry')
    elif item[6:-16]=='06' and int(item[18:-4])%2==0:
        feeling_list.append('female_fearful')
    elif item[6:-16]=='06' and int(item[18:-4])%2==1:
        feeling_list.append('male_fearful')
    elif item[6:-16]=='07' and int(item[18:-4])%2==1:
        feeling_list.append('male_disgusting')
    elif item[6:-16]=='07' and int(item[18:-4])%2==0:
        feeling_list.append('female_disgusting')
    elif item[6:-16]=='08' and int(item[18:-4])%2==1:
        feeling_list.append('male_surprise')
    elif item[6:-16]=='08' and int(item[18:-4])%2==0:
        feeling_list.append('female_surprise')
    elif item[6:-16]=='01' and int(item[18:-4])%2==1:
        feeling_list.append('male_neutral')
    elif item[6:-16]=='01' and int(item[18:-4])%2==0:
        feeling_list.append('female_neutral')


labels = pd.DataFrame(feeling_list)

##Getting the features of audio files using librosa

df = pd.DataFrame(columns=['feature'])
bookmark=0
for index,y in enumerate(mylist):
        X, sample_rate = librosa.load('C:/Users/Nouman/Desktop/Emotion-Analysis-Using-Speech-master/Actor_10/'+y, res_type='kaiser_fast',duration=5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X,
                                            sr=sample_rate,
                                            n_mfcc=13),
                        axis=0)
        feature = mfccs
        df.loc[bookmark] = [feature]
        bookmark=bookmark+1


df3 = pd.DataFrame(df['feature'].values.tolist())
newdf = pd.concat([df3,labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})

from sklearn.utils import shuffle
rnewdf = shuffle(newdf)

rnewdf=rnewdf.fillna(0)

##dividing the data into test and train
newdf1 = np.random.rand(len(rnewdf)) < 0.8
train = rnewdf[newdf1]
test = rnewdf[~newdf1]


trainfeatures = train.iloc[:, :-1]
trainlabel = train.iloc[:, -1:]
testfeatures = test.iloc[:, :-1]
testlabel = test.iloc[:, -1:]

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

## Changing dimension for CNN model

x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)

'''
model = Sequential()
model.add(lstm(128, input_shape=(216,1 )))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(12, activation='sigmoid'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])  
model.summary()

'''


model = Sequential()

model.add(Conv1D(256, 5,padding='same',
                 input_shape=(431,1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(6))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)


model.summary()


model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

cnnhistory=model.fit(x_traincnn, y_train, batch_size=32, epochs=1000, validation_data=(x_testcnn, y_test))

plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'], loc='upper right')
plt.show()

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(cnnhistory.history['accuracy'])

plt.plot(cnnhistory.history['val_accuracy'])
plt.legend(['train_accuracy', 'test_accuracy'], loc='upper left')
plt.show()

model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = os.path.join(os.getcwd(), 'C:/Users/Nouman/Desktop/Emotion-Analysis-Using-Speech-master/saved_models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

model_json = model.to_json()
with open("C:/Users/Nouman/Desktop/Emotion-Analysis-Using-Speech-master/model.json", "w") as json_file:
    json_file.write(model_json)
print("Done")

from keras.models import model_from_json

json_file = open('C:/Users/Nouman/Desktop/Emotion-Analysis-Using-Speech-master/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(
    "C:/Users/Nouman/Desktop/Emotion-Analysis-Using-Speech-master/saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

preds = loaded_model.predict(x_testcnn,
                             batch_size=32,
                             verbose=1)

preds1 = preds.argmax(axis=1)

abc = preds1.astype(int).flatten()
predictions = (lb.inverse_transform((abc)))

preddf = pd.DataFrame({'predictedvalues': predictions})

actual = y_test.argmax(axis=1)
abc123 = actual.astype(int).flatten()
actualvalues = (lb.inverse_transform((abc123)))

actualdf = pd.DataFrame({'actualvalues': actualvalues})

finaldf = actualdf.join(preddf)
finaldf.groupby('actualvalues').count()

finaldf.groupby('predictedvalues').count()
finaldf.to_csv('Predictions.csv', index=False)

import pyaudio
import wave
import speech_recognition as sr

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

RECORD_SECONDS = 5

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True, frames_per_buffer=1024)

print("**** recording")

frames = []

for i in range(0, int(44100 / 1024 * RECORD_SECONDS)):
    data = stream.read(1024)
    frames.append(data)

print("**** done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open('output.wav', 'wb')
wf.setnchannels(2)
wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wf.setframerate(44100)
wf.writeframes(b''.join(frames))
wf.close()

harvard = sr.AudioFile('output.wav')
r = sr.Recognizer()
with harvard as source:
    audio = r.record(source)
    text = r.recognize_google(audio)
try:
    print("You said: " + text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

data, sampling_rate = librosa.load('output.wav')

##% pylab inline
import os
import pandas as pd
import librosa
import glob

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

# livedf= pd.DataFrame(columns=['feature'])
X, sample_rate = librosa.load('output.wav', res_type='kaiser_fast', duration=5, sr=22050 * 2, offset=0)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
featurelive = mfccs
livedf2 = featurelive
livedf2 = pd.DataFrame(data=livedf2)

livedf2 = livedf2.stack().to_frame().T
twodim = np.expand_dims(livedf2, axis=2)
livepreds = loaded_model.predict(twodim,
                                 batch_size=32,
                                 verbose=1)

livepreds1 = livepreds.argmax(axis=1)

liveabc = livepreds1.astype(int).flatten()
print(livepreds[0])
livepredictions = (lb.inverse_transform((liveabc)))
print(livepredictions[0][0:3])

import pandas as pd

df = pd.read_csv('C:/Users/Nouman/Desktop/Emotion-Analysis-Using-Speech-master/train_data.csv')
df.head()
df.isnull().sum()
df['sentiment'].value_counts()

from sklearn.model_selection import train_test_split

X = df['content']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

text_clf.fit(X_train, y_train)
predictions = text_clf.predict(X_test)

# from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
# print(confusion_matrix(y_test,predictions))
# print(classification_report(y_test,predictions))

print('Emotion detected through words used :' + text_clf.predict([text]))
