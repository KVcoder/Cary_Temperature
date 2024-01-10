import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

x = pd.read_csv("Cary_Weather_Data.csv")
x['DATE'] = pd.to_datetime(x['YEAR'].astype(str) + '-' + x['MONTH'].astype(str) + '-' + x['DAY'].astype(str), yearfirst=True)
x = x.drop(columns=['YEAR', 'MONTH', 'DAY'])
x = x.drop(columns=['Humidity', 'Precipitation', 'Wind Speed'])
x.set_index('DATE', inplace=True)
x.head()

train_data = x.iloc[:12426, :]
valid_data = x.iloc[12426:, :]

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
valid_data = scaler.transform(valid_data)

trainX = []
trainY = []
n_future = 1
n_past = 30
for i in range(n_past, len(train_data) - n_future + 1):
    trainX.append(train_data[i - n_past:i, :])
    trainY.append(train_data[i + n_future - 1:i + n_future, 0])
trainX, trainY = np.array(trainX), np.array(trainY).reshape(-1, 1)

train_set = tf.data.Dataset.from_tensor_slices((trainX, trainY))
train_set = train_set.batch(128)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[n_past, train_data.shape[1]]))
for dilation_rate in (1, 2, 4, 8, 16, 32, 64, 128):
    model.add(
        tf.keras.layers.Conv1D(filters=64, kernel_size=8, strides=1, dilation_rate=dilation_rate, padding='causal', activation='relu')
    )
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=1))

model.compile(loss=tf.keras.losses.Huber(delta=1.0), optimizer='adam', metrics=['mae'])
tf.config.run_functions_eagerly(True)
history = model.fit(train_set, epochs=20)
