
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

PathToFiles = "input/RockCatalog/"
files = os.listdir(PathToFiles)
file_count = len(files)
Y_train = np.zeros((file_count,1))
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential # to create a cnn model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
target_size=(500, 500, 3)
area = (125, 125, 375, 375)
def prepareImages(ydataset ,datasetlength , dataset):
    X_train = np.zeros((datasetlength, 250, 250, 3))
    count = 0
    for name in os.listdir(dataset):

        exists = os.path.isfile(dataset + name + "/graphics_ext/entry_04_a.jpg")
        if(exists):
            img = image.load_img(dataset + name + "/graphics_ext/entry_04_a.jpg" , target_size)
        exists = os.path.isfile(dataset + name + "/graphics_ext/entry_04_a.gif")
        if(exists):
            img = image.load_img(dataset + name + "/graphics_ext/entry_04_a.gif" , target_size)

        classification = name[0]
        img= img.crop(area)
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train[count] = x
        ydataset[count] = classification
        count += 1
        if (count%5 == 0):
            print("Processing image: ", count+1)


    return X_train


def prepare_labels(y):
    print(y.shape)
    values=np.array(y)
    label_encoder=LabelEncoder()
    integer_encoded=label_encoder.fit_transform(values)
    print(integer_encoded)

    onehot_encoder=OneHotEncoder(sparse=False)
    integer_encoded=integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded=onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    y=onehot_encoded
    return y,label_encoder

X = prepareImages(Y_train, file_count, PathToFiles)
X /= 255

y, label_encoder = prepare_labels(Y_train)
print(y.shape)
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (9,9), padding = 'Same', activation = 'relu', input_shape = (250,250,3)))
model.add(Conv2D(filters = 16, kernel_size = (7,7), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))
model.add(Dropout(0.25))

# fully connected
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(y.shape[1], activation = "softmax"))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()
history = model.fit(X, y, epochs=24, batch_size=100, verbose=1)

plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.show()
testdir = "input/RockTest/"
test = os.listdir(testdir)
print(len(test))

Y_test = np.zeros((len(test),1))

X_test = prepareImages(Y_test, len(test), testdir)
X_test /= 255
col = ['Image']
test_df = pd.DataFrame(Y_test, columns=col)
test_df['Id'] = ''
print(test_df)
predictions = model.predict(np.array(X_test), verbose=1)
for i, pred in enumerate(predictions):
   print("Prediction #")
   print(i)
   print("Right Answer:  ")
   print(Y_test[i])
   print(pred)
   print(label_encoder.inverse_transform(pred.argsort()[-1:][::-1]))
test_df.head(10)
test_df.to_csv('submission.csv', index=False)
