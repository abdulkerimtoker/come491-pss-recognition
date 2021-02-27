import cv2

face_cascade = cv2.CascadeClassifier('C:\\Users\\toker\\PycharmProjects\\tensor\\frontal.xml')


def detect_and_write_to(path, to_path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = face_cascade.detectMultiScale(gray, 1.1, 4)[0]
    cropped = img[y:y+h, x:x+w]
    cv2.imwrite(to_path, cropped)
