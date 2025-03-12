from glob import glob
import cv2
import os
import face_recognition
import pickle


paths = glob("D:\\2025\\AI\\Face_recognition\\dataset\\*\\*")

knownEncodings = []
knownNames = []

for path in paths:
    name = os.path.basename(os.path.dirname(path))  # Lấy tên thư mục chứa ảnh
    image = cv2.imread(path)

    if image is None:
        print(f"Lỗi khi đọc ảnh: {path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image_rgb)

    if not face_locations:
        print(f"Không tìm thấy khuôn mặt trong ảnh: {path}")
        continue

    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

    for face_encoding in face_encodings:
        knownEncodings.append(face_encoding)
        knownNames.append(name)

# Lưu dữ liệu vào file pickle
data = {"encodings": knownEncodings, "names": knownNames}
save_path = "D:\\2025\\AI\\Face_recognition\\face_enc.pickle"


try:
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print("Dữ liệu đã lưu thành công vào:", save_path)
except Exception as e:
    print("Lỗi khi lưu file:", e)
