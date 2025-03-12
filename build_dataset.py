import cv2 
import os

video = cv2.VideoCapture(0)
total = 0
while True:
    ret, frame = video.read()

    cv2.imshow("video", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("k"):
        p = os.path.sep.join(["D:\2025\AI\Face_recognition\dataset\huy", "{}.png".format(str(total).zfill(5))])  
        cv2.imwrite(p, frame)
        total += 1
	# nhấn q để thoát
    elif key == ord("q"):
	    break

print("[INFO] {} face images stored".format(total))
video.release()
cv2.destroyAllWindows()