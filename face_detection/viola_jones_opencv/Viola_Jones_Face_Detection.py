# Description: This script will conduct face detection. It uses Viola Jones Classifier.

# import packages
import cv2

#==========================================
# Name: face_detector
# Purpose: detect face area in an image
# Input Parameter(s): image_name --- the image we want to do the face detection
# Return Value(s): the image with the face detection rectangle
#============================================
def face_detector(image_name):
    # load image
    img = cv2.imread(image_name)
    # load Haar Feature-based Cascade Classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # change color image into gray
    if img.shape[2] != 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # conduct face detection and draw a rectangle around face area
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (column, row, width, height) in faces:
        # draw a rectangle around face area
        img = cv2.rectangle(
            img,
            (column, row),
            (column + width, row + height),
            (0, 255, 0),
            2)

    return img


if __name__ == '__main__':
    #image_name = 'images/test.jpg'
    # image_name = 'images/ivan.jpg'
    image_name = 'images/children_family.jpg'
    img = face_detector(image_name)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass






















