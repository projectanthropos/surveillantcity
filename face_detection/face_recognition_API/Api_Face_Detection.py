# Description: This script will conduct face detection. It uses face_recognition python API.

# import package
import cv2
import face_recognition

#==========================================
# Name: face_detector
# Purpose: detect face area in an image
# Input Parameter(s): image_name --- the image we want to do the face detection
# Return Value(s): the image with the face detection rectangle
#============================================
def face_detector(image_name):
    # load image
    img = face_recognition.load_image_file(image_name)

    # conduct face detection and draw a rectangle around face area
    faces = face_recognition.face_locations(img)
    for top, right, bottom, left in faces:
        # draw a rectangle around face area
        img = cv2.rectangle(
            img,
            (left, top),
            (right, bottom),
            (0, 0, 255),
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






















