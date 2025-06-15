import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random

from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Paths
DATASET = "/home/smurfy/Desktop/Plant_Disease_Detection/DATASET/tomato/train"
DATASET2 = "/home/smurfy/Desktop/Plant_Disease_Detection/DATASET/tomato/val"

CATEGORIES = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

train_data = []
for category in CATEGORIES:
    label = CATEGORIES.index(category)
    path = os.path.join(DATASET, category)
    for img_file in os.listdir(path):
        img = cv.imread(os.path.join(path, img_file), 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (64, 64))
        train_data.append([img, label])

test_data = []
for category in CATEGORIES:
    label = CATEGORIES.index(category)
    path = os.path.join(DATASET2, category)
    for img_file in os.listdir(path):
        img = cv.imread(os.path.join(path, img_file), 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (64, 64))
        test_data.append([img, label])

random.shuffle(train_data)
random.shuffle(test_data)


X_train = [features for features, label in train_data]
y_train = [label for features, label in train_data]
X_test = [features for features, label in test_data]
y_test = [label for features, label in test_data]


label_names = [
    "BACTERIAL SPOT", "EARLY BLIGHT", "HEALTHY", "LATE BLIGHT", "LEAF MOLD",
    "SEPTORIA LEAF SPOT", "SPIDER MITE", "TARGET SPOT", "MOSAIC VIRUS", "YELLOW LEAF CURL VIRUS"
]

Y = [label_names[i] for i in y_train]
Z = [label_names[i] for i in y_test]

X_train = np.array(X_train).reshape(-1, 64, 64, 3) / 255.0
X_test = np.array(X_test).reshape(-1, 64, 64, 3) / 255.0


os.makedirs("GRAPH", exist_ok=True)

plt.figure()
ax = sns.countplot(Y, order=label_names)
ax.set_xlabel("Leaf Diseases")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
ax.set_ylabel("Image Count")
plt.tight_layout()
plt.savefig("GRAPH/train_distribution.png")
plt.close()

plt.figure()
ax = sns.countplot(Z, order=label_names)
ax.set_xlabel("Leaf Diseases")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
ax.set_ylabel("Image Count")
plt.tight_layout()
plt.savefig("GRAPH/test_distribution.png")
plt.close()


one_hot_train = to_categorical(y_train)
one_hot_test = to_categorical(y_test)


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.4))
classifier.add(Flatten())
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(10, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.summary()


hist = classifier.fit(X_train, one_hot_train, epochs=75, batch_size=128, validation_split=0.2)


classifier.save("/home/smurfy/Desktop/Plant_Disease_Detection/MAJOR_PROJECT/TOMATO/MODELS/NEW.h5")


test_loss, test_acc = classifier.evaluate(X_test, one_hot_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Classifier Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("GRAPH/loss_plot.png")
plt.close()


plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Classifier Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("GRAPH/accuracy_plot.png")
plt.close()

y_pred = np.argmax(classifier.predict(X_test), axis=1)
y_prob = classifier.predict(X_test)

fpr = {}
tpr = {}
thresh = {}
roc_auc = {}
n_class = 10

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_prob[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['orange', 'green', 'blue', 'red', 'pink', 'purple', 'brown', 'cyan', 'yellow', 'black']
for i in range(n_class):
    plt.plot(fpr[i], tpr[i], color=colors[i], label=f'{label_names[i]} AUC = {roc_auc[i]:.3f}')
plt.title('Tomato Leaves Diseases ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("GRAPH/roc_curve.png")
plt.close()

plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("GRAPH/confusion_matrix.png")
plt.close()
