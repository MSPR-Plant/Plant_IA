# MUCHAMAD RIF'AN
# 17.01.53.2021
# JARINGAN SYARAF TIRUAN 2020
# TEKNIK INFORMATIKA

import os
import fnmatch  # TAMBAHAN

from keras.preprocessing.image import ImageDataGenerator

flower_dir = '../ressources/flowers'
flower299_dir = '../ressources/flowers299'
print(os.listdir('../ressources/flowers'))
# Memasukkan direktori file

# Memasukkan librari yang dibutuhkan
# Ignore  the warnings
import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
# %matplotlib inline
style.use('fivethirtyeight')
sns.set(style='whitegrid', color_codes=True)

# model selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# dl libraraies
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D

import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2
import numpy as np
from tqdm import tqdm
import os

# 2 ) Preparing the Data
# 2.1) Making the functions to get the training and validation set from the Images

# menyiapkan variable yang memuat direktori file
X = []
Z = []
IMG_SIZE = 32
FLOWER_DAISY_DIR = flower_dir + '/daisy'
FLOWER_SUNFLOWER_DIR = flower_dir + '/sunflower'
FLOWER_TULIP_DIR = flower_dir + '/tulip'
FLOWER_DANDI_DIR = flower_dir + '/dandelion'
FLOWER_ROSE_DIR = flower_dir + '/rose'

# create a list of all the folders directory of Flowers299
FLOWER_DIR_LIST = []
for flower_NAME in os.listdir(flower299_dir):
    FLOWER_DIR_LIST.append(flower_NAME)


# membuat fungsi pelabelan file
def assign_label(img, flower_type):
    return flower_type


# konversi gambar menjadi array
def make_train_data(flower_type, DIR):
    for img in tqdm(os.listdir(DIR)):
        if fnmatch.fnmatch(img, '*.jpg'):  # TAMBAHAN
            label = assign_label(img, flower_type)
            path = os.path.join(DIR, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            X.append(np.array(img))
            Z.append(str(label))


# menyiapkan data training
make_train_data('Daisy', FLOWER_DAISY_DIR)
print(len(X))

make_train_data('Sunflower', FLOWER_SUNFLOWER_DIR)
print(len(X))

make_train_data('Tulip', FLOWER_TULIP_DIR)
print(len(X))

make_train_data('Dandelion', FLOWER_DANDI_DIR)
print(len(X))

make_train_data('Rose', FLOWER_ROSE_DIR)
print(len(X))

# for flower_type in FLOWER_DIR_LIST:
#     make_train_data(flower_type, os.path.join(flower299_dir, flower_type))
#     print(len(X))

# 2.2 ) Visualizing some Random Images
# menampilkan data
fig, ax = plt.subplots(5, 2)
fig.set_size_inches(5, 5)
for i in range(5):
    for j in range(2):
        l = rn.randint(0, len(Z))
        ax[i, j].imshow(X[l])
        ax[i, j].set_title('Flower: ' + Z[l])

plt.tight_layout()

# 2.3 ) Label Encoding the Y array (i.e. Daisy->0, Rose->1 etc...) & then One Hot Encoding
le = LabelEncoder()
Y = le.fit_transform(Z)
Y_Total = max(Y) + 1
Y = to_categorical(Y, Y_Total)
X = np.array(X)
X = X / 255

# 2.4 ) Splitting into Training and Validation Sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# 2.5 ) Setting the Random Seeds
np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

# 3 ) Modelling
# 3.1 ) Building the ConvNet Model
# # modelling starts using a CNN.
print(np.max(Y) + 1)
print(int(np.max(Y) + 1))

model = Sequential()
model.add(Conv2D(filters=1024, kernel_size=(2, 2), padding='Same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(Y_Total, activation="softmax"))

# 3.2 ) Using a LR Annealer
batch_size = 45
epochs = 10

from keras.callbacks import ReduceLROnPlateau

red_lr = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.1)

# 3.3 ) Data Augmentation to prevent Overfitting
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

# 3.4 ) Compiling the Keras Model & Summary
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 3.5 ) Fitting on the Training set and making predcitons on the Validation set
History = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(x_test, y_test),
                              verbose=1, steps_per_epoch=x_train.shape[0] // batch_size)
# model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))

model.save('../out/model_loss_{:5.2f}%_accuracy_{:5.2f}%_val_loss_{:5.2f}%_val_accuracy_{:5.2f}%.h5'.format(
    History.history['loss'][-1] * 100,
    History.history['accuracy'][-1] * 100,
    History.history['val_loss'][-1] * 100,
    History.history['val_accuracy'][-1] * 100))

from tensorflow import keras

model = keras.models.load_model('../out2/model2.h5')

model_json = model.to_json()
with open("../out2/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../out2/model.h5")
print("Saved model to disk")

# later...

# load json and create model
from keras.models import model_from_json  # TAMBAHAN

json_file = open('../out2/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../out2/model.h5")
print("Loaded model from disk")

# 4 ) Evaluating the Model Performance
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(History.history['accuracy'])  # TAMBAHAN
plt.plot(History.history['val_accuracy'])  # TAMBAHAN
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('../out2/model2.h5')

# Show the model architecture
new_model.summary()

new_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

new_model.save('../out2/model3.h5')

# 5 ) Visualizing Predictons on the Validation Set
# now storing some properly as well as misclassified indexes'.
i = 0
prop_class = []
mis_class = []
pred = model.predict(x_test)  # TAMBAHAN
pred_digits = np.argmax(pred, axis=1)  # TAMBAHAN

for i in range(len(y_test)):
    if (np.argmax(y_test[i]) == pred_digits[i]):
        prop_class.append(i)
    if (len(prop_class) == 8):
        break

i = 0
for i in range(len(y_test)):
    if (not np.argmax(y_test[i]) == pred_digits[i]):
        mis_class.append(i)
    if (len(mis_class) == 8):
        break

# CORRECTLY CLASSIFIED FLOWER IMAGES
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()

count = 0
fig, ax = plt.subplots(4, 2)
fig.set_size_inches(15, 15)
for i in range(4):
    for j in range(2):
        ax[i, j].imshow(x_test[prop_class[count]])
        ax[i, j].set_title("Predicted Flower :"
                           + str(le.inverse_transform([pred_digits[prop_class[count]]]))
                           + "\n" + "Actual Flower : "
                           + str(le.inverse_transform([np.argmax([y_test[prop_class[count]]])]))
                           )
        #         ax[i,j].set_title("Predicted Flower : "
        #                           +str(le.inverse_transform([pred_digits[prop_class[count]]]))
        #                           +"\n"+"Actual Flower : "
        #                           +str(le.inverse_transform([np.argmax([y_test[prop_class[count]]])]))
        #                          )
        plt.tight_layout()
        count += 1

# MISCLASSIFIED IMAGES OF FLOWERS
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count = 0
fig, ax = plt.subplots(4, 2)
fig.set_size_inches(15, 15)
for i in range(4):
    for j in range(2):
        ax[i, j].imshow(x_test[mis_class[count]])
        ax[i, j].set_title("Predicted Flower : "
                           + str(le.inverse_transform([pred_digits[mis_class[count]]]))
                           + "\n" + "Actual Flower : "
                           + str(le.inverse_transform([np.argmax([y_test[mis_class[count]]])])))  # tambahan []
        plt.tight_layout()
        count += 1

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('../out2/model3.h5')
# Show the model architecture
new_model.summary()

loss, acc = new_model.evaluate(x_train, y_train, verbose=2)  # y_train
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
