import csv
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model

K.set_image_dim_ordering('th')
seed = 5
np.random.seed(seed)

#load data
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

#seperate the data and labels
X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,0:1].values
X_test = df_test.iloc[:,0:].values

#reshape
X_train = X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_test = X_test.reshape(X_test.shape[0],1,28,28).astype('float32')

#normalize
X_train = X_train/255
X_test = X_test/255

#1 hot encoding
y_train=np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]

#define the CNN model
def CNN_model():
  model = Sequential()
  model.add(Conv2D(30,(5,5),input_shape=(1,28,28),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(15,(3,3),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(128,activation='relu'))
  model.add(Dense(50,activation='relu'))
  model.add(Dense(num_classes,activation='softmax'))
  
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  return model

#fit the model and save it
model = CNN_model()
model.fit(X_train,y_train,epochs=10,batch_size=100)
model.save("models/CNN_Digit_Recognizer_model_batch_100.h5")
model = load_model("models/CNN_Digit_Recognizer_model_batch_100.h5")

#use the model for prediction
y_test = [[model.predict_classes(X_test[i].reshape(1,1,28,28))[0] for i in range(X_test.shape[0])]]

#save the results into a file
fd = open("result/result_cnn_batch_100.csv",'w') 
writer = csv.writer(fd,delimiter="\n")
writer.writerows(y_test)


