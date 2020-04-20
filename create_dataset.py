import cv2
from imutils import face_utils
import dlib

id = input('Enter ID : ')
round = 0

p = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
cap = cv2.VideoCapture(0)

while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        x1, y1 = rect.left(), rect.top()
        x2, y2 = rect.right(), rect.bottom()

        x, y = x1, y1
        w, h = x2 - x1, y2 - y1

        threshold = 30
        round = round + 1
        print('Round', round)
        cv2.imwrite("dataset/User." + str(id) + "." + str(round) + ".jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(image, (x1 - threshold, y1 - threshold), (x2 + threshold, y2 + threshold), (255, 0, 0), 2)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    # show the output image with the face detections + facial landmarks

    cv2.imshow("Output", image)
    if round > 500:
         break
cv2.destroyAllWindows()
cap.release()