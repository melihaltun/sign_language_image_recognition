# @fileName signLanguageRecognition.py
# @author Melih Altun @2023

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model


from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import random
import os
import shutil

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Num GPUs Available: ', len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.chdir('./Sign-Language-Digits-Dataset')
if os.path.isdir('./train/0/') is False:
    os.mkdir('./train')
    os.mkdir('./test')
    os.mkdir('./valid')

    for i in range(0, 10):
        shutil.move(f'{i}', './train')
        os.mkdir(f'./valid/{i}')
        os.mkdir(f'./test/{i}')

        valid_samples = random.sample(os.listdir(f'./train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'./train/{i}/{j}', f'./valid/{i}')

        test_samples = random.sample(os.listdir(f'./train/{i}'), 5)
        for j in test_samples:
            shutil.move(f'./train/{i}/{j}', f'./test/{i}')

os.chdir('../')

test_path = ('./Sign-Language-Digits-Dataset/test')
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path, target_size=(224, 224), batch_size=10, shuffle=False)

#delete the model file if you want the model retrained
if os.path.isfile('models/sign_language_model.h5') is False:
    train_path = ('./Sign-Language-Digits-Dataset/train')
    valid_path = ('./Sign-Language-Digits-Dataset/valid')

    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path, target_size=(224, 224), batch_size=10)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224, 224), batch_size=10)

    assert train_batches.n == 1712
    assert valid_batches.n == 300
    assert test_batches.n == 50
    assert  test_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 10

    mobile = tf.keras.applications.mobilenet.MobileNet()

    mobile.summary()

    partialModel1 = mobile.layers[-5].output  # was [-6] on tutorial

    partialModel2 = Flatten()(partialModel1)
    output = Dense(units=10, activation='softmax')(partialModel2)

    model = Model(inputs=mobile.input, outputs = output)

    for layer in model.layers[:-23]:
        layer.trainable = False

    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_batches, validation_data=valid_batches, epochs=10) # , verbose=2)

    if os.path.isdir('./models/') is False:
        os.mkdir('./models')
    model.save('./models/sign_language_model.h5')
else:
    model = load_model('./models/sign_language_model.h5')

test_labels = test_batches.classes
predictions = model.predict(x=test_batches, verbose=0)

cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalizes Confusion Matrix")
    else:
        print("Confusion Matrix, without normalization")
    print(cm)
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

test_batches.class_indices

cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
