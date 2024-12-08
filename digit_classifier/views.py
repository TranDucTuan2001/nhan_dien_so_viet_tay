import os
import cv2
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from sklearn import svm
from skimage.feature import hog
import joblib
from sklearn.svm import LinearSVC
import tensorflow as tf

# Đường dẫn đến model đã lưu
MODEL_PATH_1 = os.path.join(os.path.dirname(__file__), 'train_model/svm_digit_classifier_1.joblib')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'train_model/svm_digit_classifier.joblib')


# Hàm tải model đã lưu (model 1)
def load_model():
    return joblib.load(MODEL_PATH_1)

# Hàm tải model đã lưu (model chính)
def load_models():
    return joblib.load(MODEL_PATH)

# Dự đoán một chữ số
def predict_digit(request):
    if request.method == 'POST' and request.FILES.get('digit_image'):
        # Lưu file được tải lên
        uploaded_file = request.FILES['digit_image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)

        # Đọc và xử lý ảnh
        img = cv2.imread(fs.path(file_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img_vector = img.reshape(1, -1)  # Chuyển đổi ảnh thành vector để dự đoán

        # Tải model và thực hiện dự đoán
        model = load_model()
        prediction = model.predict(img_vector)[0]

        # Trả về kết quả dự đoán
        return render(request, 'result.html', {
            'prediction': prediction,
            'image_url': fs.url(file_path)
        })

    # Hiển thị form tải lên nếu không phải là yêu cầu POST
    return render(request, 'upload.html')

# Xử lý ảnh chứa nhiều chữ số
def process_multiple_digits(img, model):
    # Chuyển đổi ảnh sang nhị phân
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Khởi tạo HOG descriptor cho ảnh 28x28 với cấu hình chuẩn
    hog = cv2.HOGDescriptor(_winSize=(28,28), _blockSize=(14,14), _blockStride=(14,14), _cellSize=(7,7), _nbins=9)

    for i, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        roi = img[y:y+h, x:x+w]
        roi = cv2.copyMakeBorder(roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # Tính toán đặc trưng HOG cho ROI
        roi_hog_fd = hog.compute(roi)
        
        # Kiểm tra và chuyển đổi định dạng nếu cần thiết
        if roi_hog_fd is not None and len(roi_hog_fd) == 784:
            nbr = model.predict(np.array([roi_hog_fd.flatten()], np.float32))[0]
            # Vẽ khung và chú thích chữ số dự đoán
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, str(int(nbr)), (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    # Lưu ảnh kết quả vào thư mục media
    result_path = f"{settings.MEDIA_ROOT}/result_image.jpg"
    cv2.imwrite(result_path, img)
    
    # Trả về đường dẫn ảnh kết quả
    return settings.MEDIA_URL + "result_image.jpg"

# Dự đoán nhiều chữ số trong một ảnh
def predict_multiple_digits(request):
    if request.method == 'POST' and request.FILES.get('digit_image'):
        # Lưu file được tải lên
        uploaded_file = request.FILES['digit_image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)

        # Đọc và xử lý ảnh
        image = cv2.imread(fs.path(file_path))
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)
        _, thre = cv2.threshold(im_blur, 90, 255, cv2.THRESH_BINARY_INV)

        # Tìm các đường viền cho từng chữ số
        contours, _ = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Tải model một lần để dự đoán nhiều chữ số
        model = load_models()
        predictions = []

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            roi = thre[y:y + h, x:x + w]

            # Thêm padding vào ROI với viền màu đen
            roi = np.pad(roi, ((20, 20), (20, 20)), 'constant', constant_values=(0, 0))
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

            # Tính toán đặc trưng HOG cho ROI
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), block_norm="L2")

            # Dự đoán chữ số trong ROI
            nbr = model.predict(np.array([roi_hog_fd], np.float32))[0]
            predictions.append((int(nbr), x, y))

            # Vẽ khung chữ nhật xung quanh chữ số
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Ghi chữ số dự đoán lên ảnh
            cv2.putText(image, str(int(nbr)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        # Lưu ảnh kết quả
        result_path = os.path.join(fs.location, "result_image.jpg")
        cv2.imwrite(result_path, image)

        # Trả về kết quả cho giao diện
        return render(request, 'result_multiple.html', {
            'predictions': predictions,
            'image_url': fs.url(file_path),
            'result_image_url': fs.url("result_image.jpg")  # Bao gồm URL của ảnh kết quả
        })

    # Hiển thị form tải lên nếu không phải là yêu cầu POST
    return render(request, 'upload.html')
