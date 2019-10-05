# THIS SCRIP WILL DEFINE THE FUNCTIONS IN TERMS OF FACE DETECTION AND FACE CROP
# IN PARTICULAR, HERE WE WILL USE "VIOLA_JONES FACE DETECTION ALGORITHM"

# IMPORT PACKAGES
import cv2
import os


# DEFINE A CLASS NAMED FaceDetectCrop WHICH WILL CONDUCT FACE DETECTION & CROP
class FaceDetectCrop:
    # scale_factor – Parameter specifying how much the image size is reduced at each image scale.
    # min_neighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    # min_size – Minimum possible object size. Objects smaller than that are ignored.
    # save_path_d --- THE DIRECTORY USED TO SAVE FACE DETECTION RESULT WITH RECTANGLE FACES ON IMAGES
    # save_path_c --- THE DIRECTORY USED TO SAVE CROP FACES
    # showflag --- IF TRUE, IT WILL SHOW THE FACE DETECTION IMAGE; OTHERWISE, NO FIGURES WILL SHOW
    def __init__(self, scale_factor,min_neighbors, min_size_w, min_size_h, save_path_d, save_path_c, showflag,videocropflag):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size_w = min_size_w
        self.min_size_h = min_size_h
        self.save_path_d = save_path_d
        self.save_path_c = save_path_c
        self.show = showflag
        self.videocropflag = videocropflag
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # THIS FUNCTION CONDUCTS FACE DETECTION;
    # IT WILL DRAW A RECTANGLE AROUND FACES; ALSO, CALL FACE CROP FUNCTION
    # image_name --- IS THE PROCESSING IMAGE
    def face_detector_image(self, image_name, video = False, frameid= 0):
        # load image
        if video == False:
            image = cv2.imread(image_name)
        else:
            image = image_name
        # change color image into gray
        if image.shape[2] != 1:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        face_rects = self.face_cascade.detectMultiScale(gray_image,
                                                       scaleFactor= self.scale_factor,
                                                       minNeighbors= self.min_neighbors,
                                                       minSize=(self.min_size_w, self.min_size_h))
        # NO FACES BEING DETECTED
        if len(face_rects) == 0:
            print('No faces have been detected!')
            return image

        # pt1 – Vertex of the rectangle.
        # pt2 – Vertex of the rectangle opposite to pt1 .
        # color – Rectangle color or brightness (grayscale image).
        # thickness – Thickness of lines that make up the rectangle. Negative values, like CV_FILLED , mean that the function has to draw a filled rectangle.
        # lineType – Type of the line. See the line() description.
        if video == False:
            save_name = image_name.split('/')[-1]
        face_count = 1
        for face_position in face_rects:
            x = face_position[0]
            y = face_position[1]
            w = face_position[2]
            h = face_position[3]

            # CROP FACES
            if video == False:
                save_name_c = 'result_' + str(face_count) + save_name
            else:
                save_name_c = 'v_result_' + str(frameid) + '_' + str(face_count) + '.jpg'
            final_path_c = os.path.join(self.save_path_c, save_name_c)
            self.face_crop(x, y, w, h, image, final_path_c)
            face_count += 1

            # rectangle box face
            image = cv2.rectangle(img=image,
                                  pt1=(x,y),
                                  pt2=(x+w, y+h),
                                  color=(0, 0, 255),
                                  thickness=2,
                                  lineType=8)
        # show and save image
        if video == False:
            save_name_d = 'result_' + save_name
            final_path_d = os.path.join(self.save_path_d, save_name_d)
            cv2.imwrite(final_path_d, image)
            if self.show:
                cv2.imshow('image', image)
                print('Please enter anykey to continue!')
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return image

    # THIS FUNCTION CONDUCTS FACE CROP
    # x, y, w, h --- POSITIONS WHICH RETURN BY FACE DETECTION
    # image --- THE CURRENT PROCESSING IMAGE
    # save_name --- THE NAME FOR CROPPED FACE
    def face_crop(self, x, y, w, h, image, save_name):
        if self.videocropflag == False:
            return
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)
        faceimg = image[ny:ny + nr, nx:nx + nr]
        lastimg = cv2.resize(faceimg, (32, 32))
        cv2.imwrite(save_name, lastimg)

    # video process
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#saving-a-video
    # THIS FUNCTION CONDUCTS FACE DETECTION;
    # IT WILL DRAW A RECTANGLE AROUND FACES; ALSO, CALL FACE CROP FUNCTION
    # video_name --- IS THE PROCESSING VIDEO
    def face_detector_video(self, video_name):
        save_name = video_name.split('/')[-1]
        save_name = save_name.split('.')[0]
        save_name = 'result_' + save_name + '.avi'
        final_path = os.path.join(self.save_path_d, save_name)
        input_movie = cv2.VideoCapture(video_name)
        #input_movie.open(video_name)
        fps = input_movie.get(cv2.CAP_PROP_FPS)
        frames = input_movie.get(cv2.CAP_PROP_FRAME_COUNT)
        infourcc = input_movie.get(cv2.CAP_PROP_FPS)
        framesize_w = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
        framesize_h = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        output_movie = cv2.VideoWriter(final_path, fourcc, fps, (framesize_w, framesize_h), isColor=1)
        frameid = 0
        while True:
            # Grab a single frame of video
            success, frame = input_movie.read()
            frameid += 1
            # Quit when the input video file ends
            if not success:
                break
            result_frame = self.face_detector_image (frame, video=True, frameid=frameid)
            print("Writing frame {} / {}".format(frameid, frames))
            output_movie.write(result_frame)
        # FINISH
        input_movie.release()

if __name__ == '__main__':
    pass