# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 12:05:53 2018

@author: 13 디콘 김덕영
"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import pandas as pd, numpy as np

csv = pd.read_csv("seouldata134567.csv")

pm_class ={
        "good" : [1,0,0],
        "nomal" : [0,1,0],
        "bad" : [0,0,1]
        }

#x는 입력 y는 레이블
#x= csv[["date","pm10"]].as_matrix()
y = np.empty((5000,3)) #빈 넘파이 5000개, 3줄

for i, v in enumerate(csv["label"]):
                   y[i] = pm_class[v]
                   
                   
x = csv[["date","pm10"]].as_matrix()

#모델 학습
x_train, y_train = x[1:1461], y[1:1461]
x_test, y_test = x[1461:1800], y[1461:1800]

print(len(x_train), 'date')
print(len(x_test), 'date')

#모델 만듦
model = Sequential()
model.add(Dense(1024, input_shape=(2,), init='glorot_uniform', activation='relu')) 
model.add(Dropout(0.2))

model.add(Dense(512, input_shape=(2,), activation='relu')) 
model.add(Dropout(0.2))

model.add(Dense(256, input_shape=(2,), activation='relu')) 
model.add(Dropout(0.2))

number_of_class =3
model.add(Dense(number_of_class, activation='softmax')) #출력층
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#추가 부분
history = model.fit(x_train,y_train, batch_size=500, epochs=100, validation_split=0.33)

#학습
#model.fit(x_train, y_train)
score = model.evaluate(x_test, y_test)
print()
print("score accuracy :" , score[1])

#예측, 정답 model.predict
import matplotlib.pyplot as plt
import numpy

y_acc =history.history['acc']
y_vloss = history.history['val_loss']

x_len = numpy.arange(len(y_acc))
#plt.plot(x_len,y_vloss, marker='.', c="red", markersize=3)
plt.plot(x_len,y_acc, marker='.', c="blue",  markersize=3)
plt.show()
