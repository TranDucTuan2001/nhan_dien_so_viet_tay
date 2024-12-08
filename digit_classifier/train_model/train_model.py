import os
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from tensorflow.keras.datasets import mnist
import joblib

# Đường dẫn để lưu model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'svm_digit_classifier.joblib')

def train_and_save_model():
    # Tải dữ liệu MNIST
    (X_train, y_train), _ = mnist.load_data()
    X_train = X_train.reshape(-1, 28 * 28) / 255.0  # Chuyển đổi thành vector và chuẩn hóa

    # Trích xuất đặc trưng HOG cho mỗi ảnh trong tập huấn luyện
    X_train_hog = [
        hog(img.reshape(28, 28), orientations=9, pixels_per_cell=(14, 14), 
            cells_per_block=(1, 1), block_norm="L2") 
        for img in X_train
    ]

    # Khởi tạo model SVM
    model = LinearSVC()
    model.fit(np.array(X_train_hog), y_train)  # Huấn luyện model với đặc trưng HOG và nhãn

    # Lưu model vào file
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved as '{MODEL_PATH}'")

# Gọi hàm để huấn luyện và lưu model
if __name__ == "__main__":
    train_and_save_model()
