import face_recognition
import pickle
import cv2



path = "D:\\2025\\AI\\Face_recognition\\face_enc.pickle"
data = pickle.loads(open(path, "rb").read())   


image_path= "D:\\2025\\AI\\Face_recognition\\test_images\\Test_image2.png"
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



boxes = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, boxes)

# khởi tạo list chứa tên các khuôn mặt phát hiện được
# nên nhớ trong 1 ảnh có thể phát hiện được nhiều khuôn mặt nhé
names = []

# duyệt qua các encodings của faces phát hiện được trong ảnh
for encoding in encodings:

    # trong hàm compare_faces sẽ tính Euclidean distance và so sánh với tolerance=0.6 (mặc định), nhó hơn thì khớp, ngược lại thì ko khớp (khác người)
    matches = face_recognition.compare_faces(data["encodings"], encoding, 0.4)   
    name = "Unknown"   

    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]

        counts = {}
        # duyệt qua các chỉ số được khớp và đếm số lượng 
        for i in matchedIdxs:
            name = data["names"][i]     # tên tương ứng known encoding khiowps với encoding check
            counts[name] = counts.get(name, 0) + 1  # nếu chưa có trong dict thì + 1, có rồi thì lấy số cũ + 1

        # lấy tên có nhiều counts nhất (tên có encoding khớp nhiều nhất với encoding cần check)
        name = max(counts, key=counts.get)

    names.append(name)

# Duyệt qua các bounding boxes và vẽ nó trên ảnh kèm thông tin
# Nên nhớ recognition_face trả bounding boxes ở dạng (top, rights, bottom, left)
for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15

    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

cv2.imshow("Image", image)
cv2.waitKey(0)




