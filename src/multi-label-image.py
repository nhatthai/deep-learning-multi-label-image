# Using in jumpyter notebook
# %matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from sklearn.model_selection import train_test_split


# reading the csv file
train = pd.read_csv('Multi_Label_dataset/train.csv')

# printing first five rows of the file
train.head()

train.columns

train_image = []

count = 0
max_size = 50

for i in tqdm(range(train.shape[0])):
    img = image.load_img(
        'Multi_Label_dataset/Images/' + train['Id'][i] + '.jpg',
        target_size=(400, 400, 3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)

    count += 1
    if count == max_size:
        break

X = np.array(train_image)

X.shape

plt.imshow(X[2])

# y = np.array(train.drop(['Id', 'Genre'], axis=1))
y = np.array(train.drop(['Id', 'Genre'], axis=1))[:max_size]
y.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.1)

# define model
model = Sequential()
model.add(Conv2D(
    filters=16, kernel_size=(5, 5), activation="relu",
    input_shape=(400, 400, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='sigmoid'))
model.summary()

model.compile(
    optimizer='adam', loss='binary_crossentropy',
    metrics=['accuracy'])

# training the model
model.fit(
    X_train, y_train, epochs=10,
    validation_data=(X_test, y_test), batch_size=64)

# Making Prediction for GOT.jpg
img = image.load_img('Multi_Label_dataset/GOT.jpg', target_size=(400, 400, 3))
img = image.img_to_array(img)
img = img/255
classes = np.array(train.columns[2:])
proba = model.predict(img.reshape(1, 400, 400, 3))
top_3 = np.argsort(proba[0])[:-4:-1]
for i in range(3):
    print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
