# This script isn't supposed to be run on its own, instead its accessed by the other script to train the model
# for example, in a console window, you can run the script by entering: python train.py --dataset dataset

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# This block of code declares a dataset path argument must be passed when accessing the script
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="dataset path")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="loss/accuracy plot path")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="model output path")
args = vars(ap.parse_args())

# This block of code initializes the learning rate, number of epochs, and batch size
initial_learning_rate = 1e-4
num_epochs = 20
batch_size = 32

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("Loading dataset...")
images = list(paths.list_images(args["dataset"]))
data = []
labels = []

# This block of code loops over images, gets the class labels from each image, loads and preprocesses the image, and adds the data and labels to their respective lists
for imagePath in images:
	label = imagePath.split(os.path.sep)[-2]
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)
	data.append(image)
	labels.append(label)

# This block of code converts the data and label lists to numpy.array objects, this is required since training the model requires numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# one-hot encoding of the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# This block of code splits the data into two pairs; 25% is used for testing and 75% is used for training
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# ImageDataGenerator for the manipulation and application of the mask image to the face image
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# Different model creation technique: This block of code creates a base model and a head model. The head model will be "placed" onto the base model
base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
head = base.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten(name="flatten")(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(2, activation="softmax")(head)

# Placing the head model onto the base model, now the model is trainable
model = Model(inputs=base.input, outputs=head)

# Thie block of code freezes each layer so they aren't changed during the first training iteration
for layer in base.layers:
	layer.trainable = False

# This block of code compiles the model
print("Compiling model...")
opt = Adam(lr=initial_learning_rate, decay=initial_learning_rate / num_epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# This block of code trains the head of the neural network
print("Training model...")
H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size, validation_data=(testX, testY), validation_steps=len(testX) // batch_size, epochs=num_epochs)

# This block of code is responsible for making predictions
print("Making predictions...")
predict_index = model.predict(testX, batch_size=batch_size)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predict_index = np.argmax(predict_index, axis=1)

print(classification_report(testY.argmax(axis=1), predict_index, target_names=lb.classes_))

# serializes the model and saves to file
print("Saving model...")
model.save(args["model"], save_format="h5")
print("Saved")

# plot of the training loss and the accuracy
N = num_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="training loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="training actual")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="validation actual")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
