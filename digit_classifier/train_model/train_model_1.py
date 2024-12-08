import os
import cv2
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from sklearn import svm
from skimage.feature import hog
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'svm_digit_classifier_1.joblib')

def load_model_and_save():
    model = svm.SVC(kernel='poly', C=0.1, gamma=0.0001)
    
    # Tải lên dữ liệu huấn luyện
    X_train = []
    y_train = []
    data_path = os.path.join(os.getcwd(), 'digit_classifier', 'train_ct')

    for img in os.listdir(data_path):
        filename = os.path.basename(img)
        if filename.endswith(".png"):
            image = cv2.imread(os.path.join(data_path, filename), 0)
            image = cv2.resize(image, (28, 28))
            image_vector = image.reshape(-1)
            if "_" in filename:
                label = int(filename.split("_")[0].split("so")[1])
            else:
                label = int(filename.split("so")[1].split(".png")[0])
            X_train.append(image_vector)
            y_train.append(label)
    
    # Huấn luyện mô hình
    model.fit(np.array(X_train), y_train)

    # Lưu mô hình
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved at: {MODEL_PATH}")

    return model

if __name__ == "__main__":
    load_model_and_save()
