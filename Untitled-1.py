# %%
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
x = pd.read_csv("Cary_Weather_Data.csv")
x['DATE'] = pd.to_datetime(x['YEAR'].astype(str) + '-' + x['MONTH'].astype(str) + '-' + x['DAY'].astype(str), yearfirst=True)
x = x.drop(columns=['YEAR','MONTH','DAY'])
#x = x.drop(columns=['Humidity','Precipitation','Wind Speed'])
#x.set_index('DATE', inplace=True)
x.describe().transpose()


# %%
trainDF = x[:12426]
validDF =  x[:12426]
num_features = x.shape[1]
num_features

# %%
train_mean = trainDF.mean()
train_STD = trainDF.std()

trainDF = (trainDF - train_mean) / train_STD
validDF = (validDF - train_mean) / train_STD

# %%
df_std = (x - train_mean) / train_STD
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(x.keys(), rotation=90)


# %%
class WindowGenerator():
    def __init__(self, input_width, label_width, shift, trainDF=trainDF, validDF=validDF, label_columns=None):
        
        #Store raw data
        self.trainDF = trainDF
        
        self.validDF = validDF
        
        #Figure our label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: 1 for i, name in enumerate(trainDF.columns)}
        
        # Figure out window pameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column(s): {self.label_columns}'])
#print(trainDF.head())

# %%
def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[: self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns])
    # slicing doesn't preserve static shape info
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    
    return inputs, labels

WindowGenerator.split_window = split_window

# %%
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, batch_size=32)

# %%
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1, label_columns=['Temperature']
)

# %%
w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['Precipitation'])
w2

# %%
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[5, 32]))
model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, 'linear'))

cp = tf.keras.callbacks.ModelCheckpoint('model', save_best_only=True)
#model.compile(loss='mse', optimizer='adam', metrics=['rmse'])
model.summary()

# %%
MAX_EPOCHS = 1

def compile_and_fit(model, window, patience=2): 
  #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    #patience=patience,
                                                    #mode='min')
  history = None
  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  #if (window.trainDF.empty() or window.validDF.empty() == False):
  history = model.fit(window.trainDF.to_numpy(), epochs=MAX_EPOCHS,
                      validation_data=window.validDF,
                      callbacks=[cp])
  return history

# %%
#single_step_window.trainDF.head()
#single_step_window.validDF.head()
#if ( (not single_step_window.trainDF.empty()) and (not single_step_window.validDF.empty())):
single_step_window.validDF.any()
history = compile_and_fit(model, single_step_window)


# %%
#trainX = []
#trainY = []
#n_future = 1
#n_past = 30

# %%
#def df_to_X_y(df, window_size=5):
  #df_as_np = df.to_numpy()
  #X = []
  #y = []
  #for i in range(len(df_as_np)-window_size):
    #row = [[a] for a in df_as_np[i:i+window_size]]
    #X.append(row)
    #label = df_as_np[i+window_size]
    #y.append(label)
  #return np.array(X), np.array(y)

# %%
#WINDOW_SIZE = 5
#X, y = df_to_X_y(x, WINDOW_SIZE)
#X.shape, y.shape

# %%
#X_train, y_train = X[:12426], y[:12426]
#X_val, y_val = X[12426:], y[12426:]

# %%
#for i in range(n_past, len(train_data) - n_future + 1):
    #trainX.append(train_data[i - n_past:i, 0:train_data.shape[1]])
    #trainY.append(train_data[i + n_future - 1:i +n_future, 0])
#trainX, trainY = np.array(trainX), np.array(trainY)


#train_set = tf.data.Dataset.from_tensor_slices((trainX, trainY))
#train_set = train_set.batch(128)
#trainX.shape, trainY.shape

# %%
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[5, 32]))
model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, 'linear'))

cp = tf.keras.callbacks.ModelCheckpoint('model', save_best_only=True)
model.compile(loss='mse', optimizer='adam', metrics=['rmse'])
model.summary()

X_train = tf.squeeze(X_train)
X_train = tf.transpose(X_train)

# %%
tf.config.run_functions_eagerly(True)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=cp)

# %%
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# ... (previous code)

# Making predictions on the validation split
validX = []
validY = []
for i in range(n_past, len(valid_data) - n_future + 1):
    validX.append(valid_data[i - n_past:i, 0:valid_data.shape[1]])
    validY.append(valid_data[i + n_future - 1:i + n_future, 0])
validX, validY = np.array(validX), np.array(validY)

# Reshape data for prediction
validX = validX.reshape((validX.shape[0], n_past, valid_data.shape[1]))

# Make predictions
predictions = model.predict(validX)

# Inverse transform the predictions and actual values to original scale
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
validY = scaler.inverse_transform(validY.reshape(-1, 1))

# Plotting the predictions
plt.figure(figsize=(12, 6))
plt.xlim(100, 150)
plt.plot(validY, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Validation Set: Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()


# %%
print("Visualizing the data after performing data augmentation and dropout")
print("Accuracy is being calculated")
acc = history.history['accuracy']
print("Loss is being calculated")
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(100)
print("The results are being visualized")
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


